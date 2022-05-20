import math
import os
from collections import OrderedDict
from enum import Enum
from turtle import speed

import numpy as np
import torch
from absl import app
from PIL import Image
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, JointPosition, JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete, StepDiscrete
from rlbench.backend.observation import Observation
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks.pick_and_place_test import PickAndPlaceTest
from torch.utils.data import DataLoader
from tqdm import tqdm
from workspace.learn.data import SequentialDataset
from workspace.learn.flags import FLAGS
from workspace.models.daml import Daml, load_model
from workspace.utils import get_logger


def get_obs_config() -> ObservationConfig:
    img_size = [FLAGS.im_width, FLAGS.im_height]

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    obs_config.gripper_touch_forces = False  # jaco has no touch sensors
    obs_config.record_gripper_closing = False

    # ignore all cameras except left_shoulder
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)
    obs_config.front_camera.set_all(False)

    obs_config.left_shoulder_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.left_shoulder_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False

    return obs_config


def parse_obs(obs: Observation, device):
    """rgb, depth, state"""
    rgb = obs.left_shoulder_rgb.transpose(
        2, 0, 1) / 255.0    # rgb, remember to transpose
    depth = np.expand_dims(
        obs.left_shoulder_depth, axis=0)
    if FLAGS.gripper_action:
        if FLAGS.state_size == 3:
            state = np.concatenate(
                ([obs.gripper_open], obs.gripper_pose[:3]))
        else:
            state = np.concatenate(
                (obs.joint_positions, [obs.gripper_open], obs.gripper_pose[:3]))
    else:
        if FLAGS.state_size == 3:
            state = obs.gripper_pose[:3]
        else:
            state = np.concatenate(  # remove gripper works!
                (obs.joint_positions, obs.gripper_pose[:3]))
    # skip action
    return (
        torch.tensor(np.expand_dims(rgb, 0),
                     dtype=torch.float32, device=device),
        torch.tensor(np.expand_dims(depth, 0),
                     dtype=torch.float32, device=device),
        torch.tensor(np.expand_dims(state, 0),
                     dtype=torch.float32, device=device),
    )


class Result(Enum):
    SUCCESS = 1  # success before deadline. i.e. model released the gripper
    FAILED = 2
    FORCED = 3  # success after deadline. i.e. model didn't release the gripper


def main(argv):
    save_dir = FLAGS.save_dir
    if not os.path.exists(save_dir):
        print('save dir not exist!')
        return -1
    logger = get_logger('test')
    dataset_root = FLAGS.dataset_root
    task_name = FLAGS.task_name  # should be pick_and_place_test
    if not task_name.endswith('_test'):
        print('not a test task... abort')
        return -1
    dataset_seed = FLAGS.dataset_seed
    T = FLAGS.T
    # not using batch
    adapt_lr = FLAGS.adapt_lr
    num_updates = FLAGS.num_updates
    # mdn_samples = FLAGS.mdn_samples  # not using mdn
    test_time = math.ceil(FLAGS.test_time / FLAGS.simulation_timestep)
    print(test_time)

    logger.info('test with flags: %s' % str(FLAGS.flag_values_dict()))

    # env
    obs_config = get_obs_config()
    env = Environment(
        action_mode=MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaIK(),
                                       gripper_action_mode=StepDiscrete(steps=10)),
        obs_config=obs_config,
        headless=False,
        robot_setup='jaco')
    env.launch()
    env._pyrep.set_simulation_timestep(FLAGS.simulation_timestep)
    task = env.get_task(PickAndPlaceTest)

    torch.backends.cudnn.benchmark = True

    # instead of live demo, use saved
    dataset = SequentialDataset(dataset_root, task_name, T, dataset_seed)
    dataloader = DataLoader(
        dataset, pin_memory=True, num_workers=1, persistent_workers=True)
    loader = iter(dataloader)
    dataset_len = len(dataset)

    model = Daml()
    meta_optimizer = torch.optim.Adam(model.parameters())
    load_model(model, meta_optimizer, FLAGS.iteration)

    # counters
    num_success = num_failed = num_forced = 0

    for i in tqdm(range(dataset_len)):
        h_rgb, h_depth, h_state, h_action, h_predict = [
            f.to(model.device, non_blocking=True) for f in next(loader)]

        # adapt
        model.zero_grad()
        pre_update = OrderedDict(
            model.meta_named_parameters())  # pre-update params
        batch_post_update, adapt_losses = model.adapt(
            h_rgb, h_depth, pre_update, adapt_lr, num_updates
        )  # adaptation (get post-update params)
        logger.info('%d adapt-loss %s' % (i, adapt_losses.tolist()))
        del adapt_losses

        # load task env
        v, e = dataset.get_v_e(i)
        task.set_variation(v)
        task.reset()
        obs = task.get_observation()

        pick_error = []  # record distance between predict_pose and pick target
        place_error = []  # record distance between predict_pose and place target
        gripper = []  # record gripper action

        for t in range(test_time):
            rgb, depth, state = parse_obs(obs, device=model.device)

            # action, discrete, predict_pose = model.forward(
            #     rgb, depth, state, batch_post_update[0])
            # manual forward
            conv_out = model.forward_conv(rgb, depth, batch_post_update[0])
            predict_pose = model.forward_predict_pose(
                conv_out, batch_post_update[0])
            fc_out = model.forward_fc(
                conv_out, predict_pose, state, batch_post_update[0])
            action, discrete = model.forward_action(
                fc_out, batch_post_update[0])

            tip_position = obs.gripper_pose[:3]
            end_position = action.detach().flatten().cpu().numpy()
            delta = end_position - tip_position
            distance = np.linalg.norm(delta)
            max_distance = 0.012  # restrict ||delta pos||_2
            if np.linalg.norm(delta) > (2 * np.linalg.norm(delta * max_distance / distance)):
                delta = delta * max_distance / distance
            else:
                delta *= 0.5

            target = task._task.get_fixed_orientation(tip_position + delta)

            action = np.concatenate(
                (target, discrete.detach().flatten().cpu().numpy()), axis=0)

            obs, _, terminate = task.step(action)

            predict = predict_pose.detach().flatten().cpu().numpy()
            waypoints = task._task.get_waypoints()
            pick_target = waypoints[0].get_waypoint_object().get_position()
            place_target = waypoints[2].get_waypoint_object().get_position()
            pick_error.append(np.linalg.norm(predict - pick_target))
            place_error.append(np.linalg.norm(predict - place_target))
            gripper.append(float(action[-1]))

            task._task.set_cursor_position(predict)  # display prediction pose
            task._task.set_another_cursor_position(end_position)  # display IK target pose

            del conv_out, predict_pose, fc_out, discrete

            if terminate:  # task success
                result = Result.SUCCESS
                break
            # print(obs.gripper_open)

        if not terminate:
            act = np.zeros(FLAGS.action_size)
            if FLAGS.action_size == 4:
                act = action
            act[-1] = 1.0  # release gripper
            obs, _, terminate = task.step(act)
            result = Result.FORCED if terminate else Result.FAILED

        logger.info('%d v %d e %d result %s' % (i, v, e, result))
        logger.info('%d v %d e %d pick_error %s' % (i, v, e, pick_error))
        logger.info('%d v %d e %d place_error %s' % (i, v, e, place_error))
        logger.info('%d v %d e %d gripper %s' % (i, v, e, gripper))
        num_success += 1 if result == Result.SUCCESS else 0
        num_failed += 1 if result == Result.FAILED else 0
        num_forced += 1 if result == Result.FORCED else 0
        print('success rate:', (num_success + num_forced) /
              (num_success + num_failed + num_forced))

        del pre_update, batch_post_update

    logger.info('success %d failed %d forced %d' %
                (num_success, num_failed, num_forced))

    env.shutdown()


if __name__ == '__main__':
    app.run(main)
