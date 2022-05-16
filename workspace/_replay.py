import os
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.observation import Observation
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.pick_and_place import PickAndPlace


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else os.path.join(
    os.path.expanduser('~'), 'disk/rlbench_data')

obs_config = ObservationConfig()
# obs_config.set_all(True)
obs_config.record_gripper_closing = True

env = Environment(
    action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(),
                                   gripper_action_mode=Discrete()),
    dataset_root=DATASET,
    obs_config=obs_config,
    headless=False,
    robot_setup='jaco')
env.launch()

print(env._pyrep.get_simulation_timestep())
env._pyrep.set_simulation_timestep(0.1)
print(env._pyrep.get_simulation_timestep())

task_env = env.get_task(PickAndPlace)


def fn(v=0):
    task_env.set_variation(v)
    # -> List[List[Observation]]
    demos = task_env.get_demos(
        1, live_demos, random_selection=False, from_episode_number=0)
    demo = demos[0]
    task_env.reset()

    # task = PickAndPlace()
    task = task_env._task

    init_obs = demo[0]
    task.setup_poses(init_obs.misc['poses'])

    env._pyrep.start()
    env._pyrep.step_ui()
    for observation in demo:
        # observation = Observation()
        print(observation.joint_velocities)
        print(observation.gripper_open)
        task_env.step(np.concatenate(
            [observation.joint_velocities, [observation.gripper_open]]))


if __name__ == '__main__':
    while True:
        cmd = input('cmd: ')
        if cmd == 'q':
            env.shutdown()
            break
        elif cmd == '':
            pass
        else:
            fn(int(cmd))
