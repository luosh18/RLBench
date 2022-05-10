from rlbench.tasks.pick_and_place import *


class PickAndPlaceTest(PickAndPlace):
    def __init__(self, pyrep: PyRep, robot: Robot, name: str = None):
        super().__init__(pyrep, robot, name)
        self.load_dataset_config(False)
