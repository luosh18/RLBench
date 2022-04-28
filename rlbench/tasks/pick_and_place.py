import math
import os
import time
from typing import List

import numpy as np
from pyrep.const import ObjectType
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import ConditionSet, DetectedCondition
from rlbench.backend.exceptions import WaypointError
from rlbench.backend.observation import Observation
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.waypoints import Point, PredefinedPath, Waypoint
from rlbench.const import colors

ARM_POS = np.array([-3.0895e-01, 0, +8.2002e-01])
INIT_POS = [1.461442470550537, 1.7185994386672974, 4.47493839263916,
            -0.13187171518802643, -2.4193673133850098, 0.6419594287872314]


def _fix_orientation(point: Object) -> np.ndarray:
    # delta = (\Delta_x, \Delta_y)
    delta = point.get_position()[:2] - ARM_POS[:2]
    distance = np.linalg.norm(delta)
    rotation = -math.atan(delta[1] / delta[0])
    rotation_fix = 4 * math.pi * ((max(distance, 0.55) - 0.55) ** 2)
    point.set_orientation(
        [math.radians(-90), rotation + rotation_fix, math.radians(-90)])
    return point.get_orientation()


def fix_orientation(waypoint: Point) -> np.ndarray:
    return _fix_orientation(waypoint.get_waypoint_object())


class PickAndPlace(Task):

    def init_task(self) -> None:
        self.pick_dummy = Dummy('pick_dummy')
        self.place_dummy = Dummy('place_dummy')
        self.distractor_dummies = [
            Dummy('distractor_dummy%d' % i) for i in range(2)]
        self.pick_boundary = SpawnBoundary([Shape('pick_boundary')])
        self.place_boundary = SpawnBoundary([Shape('place_boundary')])
        self.success_detector = ProximitySensor('place_success')

        [self.register_waypoint_ability_start(i, fix_orientation) for i in range(20)]

        self.spawned_objects: List[Shape] = []

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        ARM_POS = self.robot.arm.get_position()
        self.robot.arm.set_joint_positions(INIT_POS, True)
        self.setup_objects(OBJ_LIST[index])
        # set color. TODO: set texture instead
        np.random.seed(index)
        color_choices = np.random.choice(list(range(len(colors))), size=4, replace=False)
        for color_choice, spawned_object in zip(color_choices, self.spawned_objects):
            _, rgb = colors[color_choice]
            [o.set_color(rgb) for o in spawned_object.get_objects_in_tree(object_type=ObjectType.SHAPE)]
        np.random.seed()
        pick_color_name, _ = colors[color_choices[0]]
        place_color_name, _ = colors[color_choices[1]]

        return [pick_color_name, place_color_name, self.setup_poses()]

    def variation_count(self) -> int:
        return len(OBJ_LIST)

    def is_static_workspace(self) -> bool:
        return True

    def cleanup(self) -> None:
        self.success_detector.set_parent(self.place_dummy)
        [obj.remove() for obj in self.spawned_objects if obj.still_exists()]
        self.spawned_objects.clear()

    def decorate_observation(self, observation: Observation) -> Observation:
        poses = [self.pick_dummy.get_pose(),
                 self.place_dummy.get_pose(),
                 self.distractor_dummies[0].get_pose(),
                 self.distractor_dummies[1].get_pose()]
        observation.misc['poses'] = poses
        return observation

    def spawn_object(self, name, parent_object: Object = None) -> Shape:
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../assets/pick_and_place_ttms')
        filename = os.path.join(assets_dir, name + '.ttm')
        obj = self.pyrep.import_model(filename)
        if parent_object is not None:
            obj.set_parent(parent_object, False)
            obj.set_position(np.zeros(3), relative_to=parent_object, reset_dynamics=True)
        _, _, _, _, min_z, _ = obj.get_bounding_box()
        x, y, z = obj.get_position()
        obj.set_position([x, y, z - min_z])
        self.spawned_objects.append(obj)
        return obj

    def setup_objects(self, object_names: List[str], texture_names: List[str] = None):
        self.object_names = object_names
        self.texture_names = texture_names
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
        self.register_success_conditions(
            [DetectedCondition(self.pick_target, self.success_detector)])
        # set textures
        if texture_names is not None and len(texture_names) == len(object_names):
            for obj, texture in zip(self.spawned_objects, texture_names):
                # load and set texture
                pass

    def setup_poses(self, poses: List[np.ndarray] = None) -> List[np.ndarray]:
        if poses is None:
            # randonmize positions
            self.pick_boundary.clear()
            self.pick_boundary.sample(self.pick_dummy)
            self.place_boundary.clear()
            for dummy in [self.place_dummy] + self.distractor_dummies:
                self.place_boundary.sample(dummy, min_distance=0.05)
        else:
            for pose, dummy in zip(poses, [self.pick_dummy, self.place_dummy] + self.distractor_dummies):
                dummy.set_pose(pose)
        self.object_poses = [obj.get_pose() for obj in [self.pick_dummy, self.place_dummy] + self.distractor_dummies]
        return self.object_poses

    # def get_setup_and_demo(self):
    #     return SetupAndDemo(self.object_names, self.texture_names, self.object_poses)

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
        [fix_orientation(p) for p in waypoints]
        # Check if all of the waypoints are feasible
        feasible, way_i = self._feasible(waypoints)
        if not feasible:
            raise WaypointError(
                "Infeasible episode. Can't reach waypoint %d." % way_i, self)
        for func, way in additional_waypoint_inits:
            func(way)
        return waypoints


OBJ_LIST = [
    ['Knight', 'Low_poly_bowl_or_cup', 'plate', 'plate']
]
