import os
import pickle
from multiprocessing import Manager, Process

import numpy as np
import rlbench.backend.task as task
from absl import app, flags
from PIL import Image
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.backend.observation import Observation
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_float('simulation_timestep', 0.1,  # TODO: remember to set simulation_timestep while testing
                   'default 0.1 second for each frame')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return False
    return True


def get_obs_config() -> ObservationConfig:
    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    obs_config.gripper_touch_forces = False  # jaco has no touch sensors
    obs_config.record_gripper_closing = True  # new dataset

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

    return obs_config


def process_demo(demo):
    # get needed data
    contact_frame = 0
    rgb, depth, state, action = [], [], [], []
    waypoint = []
    gripper_action = []
    gif_frames = []
    # for each observation (frame) (sliding window: 2)
    for i, obs in enumerate(demo[1:]):
        pre_obs: Observation = demo[i]
        rgb.append(pre_obs.left_shoulder_rgb.transpose(
            2, 0, 1))  # rgb, remember to transpose
        depth.append(np.expand_dims(
            pre_obs.left_shoulder_depth, axis=0))  # depth
        state.append(  # state (joint position, gipper_open, gripper 'position')
            np.concatenate((pre_obs.joint_positions, [pre_obs.gripper_open], pre_obs.gripper_pose[:3])))
        action.append(  # action (joint velocity, gripper_open)
            np.concatenate((obs.joint_velocities, [obs.gripper_open])))
        waypoint.append(obs.misc['waypoint_pose'])
        gripper_action.append(
            np.array([pre_obs.misc['gripper_action']]))
        gif_frames.append(Image.fromarray(pre_obs.left_shoulder_rgb))
        # find the frame when gripper contact the target obj
        if (not contact_frame > 0) and (obs.gripper_open == 0.0):
            contact_frame = i  # the frame before gripper closed
    contact_pos = demo[contact_frame].gripper_pose[:3]
    end_pos = demo[-1].gripper_pose[:3]
    predict_pos = [contact_pos if i <= contact_frame else end_pos
                   for i in range(len(rgb))]
    processed_demo = {
        'rgb': np.array(rgb),
        'depth': np.array(depth),
        'state': np.array(state),
        'action': np.array(action),
        'predict': np.array(predict_pos),
        'waypoint': np.array(waypoint),
        'gripper_action': np.array(gripper_action),
    }
    # for name in processed_demo:
    #     print(name,  processed_demo[name][contact_frame].shape)
    #     print(processed_demo[name][contact_frame])
    return processed_demo, gif_frames


def save_demo(processed_demo, gif_frames, example_path):
    # Save the low-dimension data
    with open(os.path.join(example_path, 'demo.pkl'), 'wb') as f:
        pickle.dump(processed_demo, f)
    gif_frames[0].save(os.path.join(example_path, 'ref.gif'),
                       save_all=True, append_images=gif_frames[1:],
                       optimize=False, duration=FLAGS.simulation_timestep*1000, loop=0)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=get_obs_config(),
        headless=True,
        robot_setup='jaco')
    rlbench_env.launch()

    rlbench_env._pyrep.set_simulation_timestep(FLAGS.simulation_timestep)

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, obs = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        check_and_make(variation_path)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
            with file_lock:
                if check_and_make(episode_path):
                    print('Process', i, '// Task:', task_env.get_name(),
                          '// Variation:', my_variation_count, '// Demo:', ex_idx,
                          'exist, skip...')
                    continue
                else:
                    print('Process', i, '// Task:', task_env.get_name(),
                          '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                processed_demo, gif_frames = process_demo(demo)
                with file_lock:
                    save_demo(processed_demo, gif_frames, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]
    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
    app.run(main)
