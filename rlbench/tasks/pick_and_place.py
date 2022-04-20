import math
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
        self.pick_target = Shape('pick_target')
        self.place_dummy = Dummy('place_dummy')
        self.place_target = Shape('place_target')
        # self.distractors = [
        #     Shape('stack_blocks_distractor%d' % i)
        #     for i in range(2)]
        self.register_graspable_objects([self.pick_target])
        self.pick_boundary = SpawnBoundary([Shape('pick_boundary')])
        self.place_boundary = SpawnBoundary([Shape('place_boundary')])
        self.success_detector = ProximitySensor('place_success')

        cond_set = ConditionSet([
            # GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.pick_target, self.success_detector)
        ])
        self.register_success_conditions([cond_set])
        # add fix_orientation for all waypoints
        [self.register_waypoint_ability_start(i, fix_orientation) for i in range(20)]

    def init_episode(self, index: int) -> List[str]:
        ARM_POS = self.robot.arm.get_position()
        self.robot.arm.set_joint_positions(INIT_POS, True)
        pick_color_name, pick_rgb = colors[index]
        self.pick_target.set_color(pick_rgb)

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        place_color_name, place_rgb = colors[color_choices[0]]
        self.place_target.set_color(place_rgb)

        self.place_boundary.clear()
        self.place_boundary.sample(
            self.place_dummy, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        self.pick_boundary.clear()
        self.pick_boundary.sample(self.pick_dummy)
        # for block in [self.target_block] + self.distractors:
        #     self.boundary.sample(block, min_distance=0.1)

        return ['pick up the %s object and place it to the target' % pick_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def is_static_workspace(self) -> bool:
        return True

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
