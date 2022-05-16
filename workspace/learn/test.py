import os
from collections import OrderedDict

import numpy as np
import torch
from absl import app
from PIL import Image
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
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
    state = np.concatenate(
        (obs.joint_positions, [obs.gripper_open], obs.gripper_pose[:3]))
    # skip action
    # DEBUG
    # img = rgb.transpose(1, 2, 0) * 255
    # print(img.shape)
    # img = Image.fromarray(np.uint8(img))
    # img.show()
    # input('parse_obs')
    return (
        torch.tensor(np.expand_dims(rgb, 0),
                     dtype=torch.float32, device=device),
        torch.tensor(np.expand_dims(depth, 0),
                     dtype=torch.float32, device=device),
        torch.tensor(np.expand_dims(state, 0),
                     dtype=torch.float32, device=device),
    )


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
    mdn_samples = FLAGS.mdn_samples

    logger.info('test with flags: %s' % str(FLAGS.flag_values_dict()))

    # env
    obs_config = get_obs_config()
    env = Environment(
        action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(),
                                       gripper_action_mode=Discrete()),
        obs_config=obs_config,
        headless=False,
        robot_setup='jaco')
    env.launch()
    print(env._pyrep.get_simulation_timestep())
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

    for i in tqdm(range(10)):
        h_rgb, h_depth, h_state, h_action, h_predict = [
            f.to(model.device, non_blocking=True) for f in next(loader)]

        # DEBUG
        # img = h_rgb[0][0].clone().detach().cpu().numpy().transpose(1, 2, 0) * 255
        # print(img.shape)
        # img = Image.fromarray(np.uint8(img))
        # img.show()
        # input()
        # raise RuntimeError

        model.zero_grad()

        # adapt
        pre_update = OrderedDict(
            model.meta_named_parameters())  # pre-update params
        batch_post_update, adapt_losses = model.adapt(
            h_rgb, h_depth, pre_update, adapt_lr, num_updates
        )  # adaptation (get post-update params)
        logger.info('%d adapt-loss %s' % (
            i, adapt_losses.tolist()))

        # load task env
        v, e = dataset.get_v_e(i)
        task.set_variation(v)
        task.reset()
        # x = input('step_ui')
        # env._robot.arm.set_joint_positions(
        #     [1.0296657085418701, 1.869177222251892, 4.730264663696289, -
        #         0.334658145904541, -1.8943476676940918, 0.22506999969482422],
        #     True
        # )

        # DEBUG
        # for action in h_action[0]:
        #     print(action.shape)
        #     task.step(action.cpu().detach().flatten().numpy())
        # for state in h_state[0]:
        #     task._robot.arm.set_joint_positions(state[:6].tolist(), True)
        #     task._pyrep.step()

        # test
        obs = task.get_observation()
        for t in range(200):  # hard-coded teting time 10s
            rgb, depth, state = parse_obs(obs, device=model.device)
            # print(rgb.shape, depth.shape, state.shape, sep='\n')

            # action, discrete, predict_pose = model.forward(
            #     rgb, depth, state, batch_post_update[0])

            conv_out = model.forward_conv(rgb, depth, batch_post_update[0])
            predict_pose = model.forward_predict_pose(conv_out, batch_post_update[0])
            fc_out = model.forward_fc(conv_out, predict_pose, state, batch_post_update[0])
            action, discrete = model.forward_action(fc_out, batch_post_update[0])

            action = torch.cat(
                (action, discrete), dim=1
            ).flatten().cpu().detach().numpy()
            print(t, action, predict_pose.flatten().cpu().detach().numpy())
            # input()
            if action[-1] < 0.90:
                action[-1] = 0.0
            elif action[-1] > 0.3:
                action[-1] = 1.0
            # print(action.shape, action)

            # wps = task._task.get_waypoints()
            # action = wps[0].get_waypoint_object().get_pose()
            # action = np.append(action, 1.0)

            obs, _, terminate = task.step(action)
            if terminate:
                print(terminate)
                break
            # print(obs.gripper_open)

        raise RuntimeError

    env.shutdown()


if __name__ == '__main__':
    app.run(main)
