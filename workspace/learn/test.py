import os
from collections import OrderedDict

import torch
from absl import app
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from torch.utils.data import DataLoader
from tqdm import tqdm
from workspace.learn.data import SequentialDataset
from workspace.learn.flags import FLAGS
from workspace.models.daml import Daml
from workspace.utils import get_logger
from rlbench.tasks.pick_and_place_test import PickAndPlaceTest


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

    obs_config.right_shoulder_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False

    return obs_config


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
    task = env.get_task(PickAndPlaceTest)

    torch.backends.cudnn.benchmark = True

    # instead of live demo, use saved
    dataset = SequentialDataset(dataset_root, task_name, T, dataset_seed)
    dataloader = DataLoader(
        dataset, pin_memory=True, num_workers=1, persistent_workers=True)
    loader = iter(dataloader)
    dataset_len = len(dataset)

    model = Daml()

    for i in tqdm(range(500, 510)):
        h_rgb, h_depth, _, _, _ = [
            f.to(model.device, non_blocking=True) for f in next(loader)]

        model.zero_grad()

        # # adapt
        # pre_update = OrderedDict(
        #     model.meta_named_parameters())  # pre-update params
        # batch_post_update, adapt_losses = model.adapt(
        #     h_rgb, h_depth, pre_update, adapt_lr, num_updates
        # )  # adaptation (get post-update params)
        # logger.info('%d adapt-loss %s' % (
        #     i, adapt_losses.tolist()))

        # TODO testing (set variation, test, log)
        v, e = dataset.get_v_e(i)
        task.set_variation(v)
        task.reset()
        x = input('step_ui')

    env.shutdown()

if __name__ == '__main__':
    app.run(main)
