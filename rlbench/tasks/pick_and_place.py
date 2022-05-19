import math
import os
import pickle
import time
from typing import List

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType, TextureMappingMode
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.robots.end_effectors.gripper import CLOSING, OPENING
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.exceptions import WaypointError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Point, PredefinedPath, Waypoint
from rlbench.const import colors


class DatasetConfig():
    def __init__(self, VARIATION, EPISODE, TEST_VARIATION, TEST_EPISODE,
                 train_objects: List[List[str]],
                 train_textures: List[List[str]],
                 test_objects: List[List[str]],
                 test_textures: List[List[str]],) -> None:
        assert VARIATION == len(train_objects) == len(train_textures)
        assert TEST_VARIATION == len(test_objects) == len(test_textures)
        self.VARIATION = VARIATION
        self.EPISODE = EPISODE
        self.TEST_VARIATION = TEST_VARIATION
        self.TEST_EPISODE = TEST_EPISODE
        self.train_objects = train_objects
        self.train_textures = train_textures
        self.test_objects = test_objects
        self.test_textures = test_textures

    def __str__(self) -> str:
        return str(self.__dict__)


def _fix_orientation(point: Object) -> np.ndarray:
    # delta = (\Delta_x, \Delta_y)
    delta = point.get_position()[:2] - PickAndPlace.ARM_POS[:2]
    distance = np.linalg.norm(delta)
    rotation = -math.atan(delta[1] / delta[0])
    rotation_fix = 4 * math.pi * ((max(distance, 0.55) - 0.55) ** 2)
    point.set_orientation(
        [math.radians(-90), rotation + rotation_fix, math.radians(-90)])
    point.rotate([0, math.radians(-12), 0])
    return point.get_pose()

def fix_waypoint(waypoint: Point, update=True) -> np.ndarray:
    pose = _fix_orientation(waypoint.get_waypoint_object())
    if update:
        PickAndPlace.WAYPOINT_POSE = pose
    return pose


class PickAndPlace(Task):
    ARM_POS = np.array([-3.0895e-01, 0, +8.2002e-01])
    INIT_POS = [1.461442470550537, 1.7185994386672974, 4.47493839263916,
                -0.13187171518802643, -2.4193673133850098, 0.6419594287872314]
    WAYPOINT_POSE = np.zeros(7)

    def __init__(self, pyrep: PyRep, robot: Robot, name: str = None):
        super().__init__(pyrep, robot, name)
        self._loaded_dataset_config = False

    def init_task(self) -> None:
        self.useful = Dummy('useful')  # for variation display
        self.cursor = Shape('cursor')  # for predict target pose display
        self.pick_dummy = Dummy('pick_dummy')
        self.place_dummy = Dummy('place_dummy')
        self.distractor_dummies = [
            Dummy('distractor_dummy%d' % i) for i in range(2)]
        self.pick_boundary = SpawnBoundary([Shape('pick_boundary')])
        self.place_boundary = SpawnBoundary([Shape('place_boundary')])
        self.success_detector = ProximitySensor('place_success')

        for i in range(20):
            self.register_waypoint_ability_start(i, fix_waypoint)

        self.spawned_objects: List[Shape] = []

        if not self._loaded_dataset_config:
            self.load_dataset_config()

    def load_dataset_config(self, train=True):
        """default load training set"""
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../assets/dataset_config.pkl')
        with open(filename, 'rb') as f:
            dataset_config = pickle.load(f)
            self.OBJ_LIST = dataset_config.train_objects if train else dataset_config.test_objects
            self.TEX_LIST = dataset_config.train_textures if train else dataset_config.test_textures
            self._loaded_dataset_config = True

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        PickAndPlace.ARM_POS = self.robot.arm.get_position()
        self.robot.arm.set_joint_positions(PickAndPlace.INIT_POS, True)
        self.setup_objects(self.OBJ_LIST[index])
        self.setup_textures(self.TEX_LIST[index])
        self.setup_poses()
        self.useful.set_name('variation_%d' % index)
        return self.OBJ_LIST[index] + self.TEX_LIST[index]

    def variation_count(self) -> int:
        if not self._loaded_dataset_config:
            self.load_dataset_config()
        return len(self.OBJ_LIST)

    def is_static_workspace(self) -> bool:
        return True

    def cleanup(self) -> None:
        self.useful.set_name('useful')
        self.success_detector.set_parent(self.place_dummy)
        [obj.remove() for obj in self.spawned_objects if obj.still_exists()]
        self.spawned_objects.clear()

    def decorate_observation(self, observation: Observation) -> Observation:
        if self.robot.gripper.action == CLOSING:
            observation.gripper_open = (  # object attached then set to 0
                CLOSING if len(self.robot.gripper.get_grasped_objects()) > 0
                else OPENING)
        poses = [self.pick_dummy.get_pose(),
                 self.place_dummy.get_pose(),
                 self.distractor_dummies[0].get_pose(),
                 self.distractor_dummies[1].get_pose()]
        observation.misc['poses'] = poses
        observation.misc['waypoint_pose'] = PickAndPlace.WAYPOINT_POSE
        observation.misc['gripper_action'] = self.robot.gripper.action
        return observation

    def get_low_dim_state(self) -> np.ndarray:
        return np.array([obj.get_pose() for obj in self.spawned_objects]).flatten()

    def spawn_object(self, name, parent_object: Object = None) -> Shape:
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../assets/pick_and_place_ttms')
        filename = os.path.join(assets_dir, name + '.ttm')
        obj = self.pyrep.import_model(filename)
        if parent_object is not None:
            obj.set_parent(parent_object, False)
            obj.set_position(np.zeros(3), parent_object, True)
            _, _, _, _, min_z, _ = obj.get_bounding_box()
            x, y, z = parent_object.get_position()
            parent_object.set_position([x, y, z - min_z])
        self.spawned_objects.append(obj)
        return obj

    def setup_objects(self, object_names: List[str]):
        self.object_names = object_names
        # spawn objects
        self.cleanup()
        for name, parent_object in zip(object_names, [self.pick_dummy, self.place_dummy] + self.distractor_dummies):
            self.spawn_object(name, parent_object)
        self.pick_target = self.spawned_objects[0]
        self.place_target = self.spawned_objects[1]
        self.distractors = self.spawned_objects[2:]
        # set success_detector's position and parent
        self.success_detector.set_parent(self.place_target)
        self.success_detector.set_position(
            self.place_target.get_position())
        # register success conditions
        self.register_graspable_objects([self.pick_target])
        self.register_success_conditions([
            DetectedCondition(self.pick_target, self.success_detector),
            NothingGrasped(self.robot.gripper),
        ])

    def setup_textures(self, texture_names: List[str]):
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../assets/textures')
        if len(texture_names) == len(self.spawned_objects):
            self.texture_names = texture_names
            for obj, name in zip(self.spawned_objects, texture_names):
                filename = os.path.join(assets_dir, str(name) + '.png')
                text_ob, texture = self.pyrep.create_texture(filename)
                for shape in obj.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False):
                    if shape.is_renderable():
                        shape.set_texture(texture, TextureMappingMode.CUBE,
                                          repeat_along_u=True, repeat_along_v=True,
                                          uv_scaling=[.15, .15])
                text_ob.remove()

    def setup_poses(self, poses: List[np.ndarray] = None) -> List[np.ndarray]:
        if poses is None:
            # randonmize positions
            self.pick_boundary.clear()
            self.pick_boundary.sample(self.pick_dummy)
            self.place_boundary.clear()
            for dummy in [self.place_dummy] + self.distractor_dummies:
                self.place_boundary.sample(dummy, min_distance=0.14)
        else:
            for pose, dummy in zip(poses, [self.pick_dummy, self.place_dummy] + self.distractor_dummies):
                dummy.set_pose(pose)
        self.object_poses = [obj.get_pose() for obj in
                             [self.pick_dummy, self.place_dummy] + self.distractor_dummies]
        return self.object_poses

    def get_fixed_orientation(self, pos: np.ndarray) -> np.ndarray:
        pt = self.useful
        pt.set_position(pos)
        pose = _fix_orientation(pt)
        return pose

    def set_cursor_position(self, pos: np.ndarray):
        self.cursor.set_position(pos)

    def _get_waypoints(self, validating=False) -> List[Waypoint]:
        waypoint_name = 'waypoint%d'
        waypoints = []
        additional_waypoint_inits = []
        i = 0
        while True:
            name = waypoint_name % i
            if not Object.exists(name) or i == self._stop_at_waypoint_index:
                # There are no more waypoints...
                break
            ob_type = Object.get_object_type(name)
            way = None
            if ob_type == ObjectType.DUMMY:
                waypoint = Dummy(name)
                start_func = None
                end_func = None
                if i in self._waypoint_abilities_start:
                    start_func = self._waypoint_abilities_start[i]
                if i in self._waypoint_abilities_end:
                    end_func = self._waypoint_abilities_end[i]
                way = Point(waypoint, self.robot,
                            start_of_path_func=start_func,
                            end_of_path_func=end_func)
            elif ob_type == ObjectType.PATH:
                cartestian_path = CartesianPath(name)
                way = PredefinedPath(cartestian_path, self.robot)
            else:
                raise WaypointError(
                    '%s is an unsupported waypoint type %s' % (
                        name, ob_type), self)

            if name in self._waypoint_additional_inits and not validating:
                additional_waypoint_inits.append(
                    (self._waypoint_additional_inits[name], way))
            waypoints.append(way)
            i += 1

        # fix_orientation before check feasible
        [fix_waypoint(p) for p in reversed(waypoints)]
        # Check if all of the waypoints are feasible
        feasible, way_i = self._feasible(waypoints)
        if not feasible:
            raise WaypointError(
                "Infeasible episode. Can't reach waypoint %d." % way_i, self)
        for func, way in additional_waypoint_inits:
            func(way)
        return waypoints


# OBJ_LIST = [
#     ['targets/Knight', 'bowls/Low_poly_bowl_or_cup', 'plates/-dinnerplate--148425', 'plates/plate'],
#     ['targets/Knight', 'plates/-dinnerplate--148425', 'plates/plate--81292', 'plates/plate'],
#     ['targets/Knight', 'plates/plate--81292', 'plates/russian-porcelain-plate-free-3d-model-98095', 'plates/plate'],
#     ['targets/Knight', 'plates/russian-porcelain-plate-free-3d-model-98095', 'plates/-dinnerplate--148425', 'plates/plate'],
#     ['targets/Knight', 'plates/spinning-plate-v1--644338', 'plates/-dinnerplate--148425', 'plates/plate'],
# ]

# TEX_LIST = [
#     ['banded_0002', 'banded_0013', 'banded_0046', 'banded_0077'],
#     ['banded_0117', 'banded_0128', 'blotchy_0003', 'blotchy_0017'],
#     ['banded_0036', 'banded_0090', 'lined_0086', 'banded_0067'],
#     ['chequered_0066', 'banded_0092', 'chequered_0066', 'cobwebbed_0053'],
#     ['chequered_0066', 'banded_0092', 'chequered_0066', 'cobwebbed_0053'],
# ]
