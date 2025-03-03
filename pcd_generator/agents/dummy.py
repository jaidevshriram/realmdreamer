from .base import Agent

from rich import print
import numpy as np


class DummyAgent(Agent):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.poses_list = np.load(cfg.poses_path)
        print("Using dummy agent with preloaded poses from ", cfg.poses_path)

    def load_poses(self, poses_path):
        self.poses_list = np.load(poses_path)
        print("Loaded poses from", poses_path)

    def get_next_element(self):
        """Returns the ith pose from the list of poses."""

        if self._i < len(self.poses_list):
            return self.poses_list[self._i]
        else:
            return None
