from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

INIT_POS = [1.2, 1.8788, 4.4146, -0.0732, -2.6017, 0.5632]


class TutoSlideBlockToTarget(Task):

    def init_task(self) -> None:
        self.robot.arm.set_joint_positions(INIT_POS, True)
        self.block = Shape('block')
        success_detector = ProximitySensor('success')
        self.target = Shape('target')
        self.boundary = SpawnBoundary([Shape('boundary')])
        success_condition = DetectedCondition(self.block, success_detector)
        self.register_success_conditions([success_condition])
        pass

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self.robot.arm.set_joint_positions(INIT_POS, True)
        block_color_name, block_rgb = colors[index]
        self.block.set_color(block_rgb)
        self.boundary.clear()
        self.boundary.sample(self.target)
        return ['slide the %s block to target' % block_color_name,
                'push the %s cube to the green plane' % block_color_name,
                'nudge the %s block so that it covers the white target' % block_color_name,
                'Find the %s item on the table and manipulate its position so that it reaches the plane' % block_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass

    def is_static_workspace(self) -> bool:
        return True