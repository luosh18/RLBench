import os
from multiprocessing import Manager, Process
import pickle

import numpy as np
from absl import app, flags
from pyrep.const import RenderMode
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.const import *
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.pick_and_place import PickAndPlace
from tools.cinematic_recorder import FLAGS

FLGAS = flags.FLAGS
flags.DEFINE_bool('live_demos', False,
                  'use live demos or saved demos from "data_path"')
flags.DEFINE_string('data_path', '/home/ubuntu/disk/rlbench_data',
                    'Where to load the saved raw demos.')
flags.DEFINE_string('save_path',
                    '/home/ubuntu/disk/preprocessed_data/',
                    'Where to save the preprocessed datasets.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')


def check_and_make(dir) -> bool:
    if not os.path.exists(dir):
        os.makedirs(dir)
        return False
    return True


def get_obs_config() -> ObservationConfig:
    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    obs_config.gripper_touch_forces = False  # jaco has no touch sensors
    # to be determined, Gripper.actuate might need some time
    obs_config.record_gripper_closing = False

    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    return obs_config


def process_demo(demo: Demo):
    # get needed data
    contact_frame = 0
    rgb, depth, state, action = [], [], [], []
    # for each observation (frame) (sliding window: 2)
    for i, obs in enumerate(demo[1:]):
        pre_obs: Observation = demo[i]
        rgb.append(pre_obs.left_shoulder_rgb.transpose(
            2, 0, 1))  # rgb, remember to transpose
        depth.append(np.expand_dims(
            pre_obs.left_shoulder_depth, axis=0))  # depth
        state.append(  # state (joint position, gripper 'position', gipper_open)
            np.concatenate((pre_obs.joint_positions, pre_obs.gripper_pose[:3], [pre_obs.gripper_open])))
        action.append(  # action (joint velocity, gripper_open)
            np.concatenate((obs.joint_velocities, [obs.gripper_open])))
        # find the frame when gripper contact the target obj
        if (not contact_frame > 0) and (obs.gripper_open == 0.0):
            contact_frame = i  # the frame before gripper closed
    contact_pos = demo[contact_frame].gripper_pose[:3]
    end_pos = demo[-1].gripper_pose[:3]
    predict_pos = [contact_pos if i <= contact_frame else end_pos
                   for i in range(len(rgb))]
    processed = {
        'rgb': np.array(rgb),
        'depth': np.array(depth),
        'state': np.array(state),
        'action': np.array(action),
        'predict': np.array(predict_pos),
    }
    return processed


def main(argv):
    check_and_make(FLAGS.save_path)

    ##### setup env #####
    live_demos = FLAGS.live_demos
    DATASET = '' if live_demos else FLGAS.data_path

    env = Environment(
        action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(),
                                       gripper_action_mode=Discrete()),
        dataset_root=DATASET,
        obs_config=get_obs_config(),
        headless=True,
        robot_setup='jaco')
    env.launch()

    task_env = env.get_task(PickAndPlace)
    # var_target = task_env.variation_count()
    var_target = 2

    for v in range(var_target):
        task_env.set_variation(v)
        descriptions, obs = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % v)

        check_and_make(variation_path)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        # load all demonstration
        demos = task_env.get_demos(2, live_demos, random_selection=False)
        print('variation: %d has %d demos' % (v, len(demos)))
        for i, demo in enumerate(demos):  # for each demo
            parsed = process_demo(demo)
            demo_path = os.path.join(variation_path, EPISODE_FOLDER % i + '.pkl')
            with open(demo_path, 'wb') as f:
                print('demo %d parsed, save to %s' % (i, demo_path))
                pickle.dump(parsed, f)

    env.shutdown()


if __name__ == '__main__':
    app.run(main)
