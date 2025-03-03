from abc import ABC, abstractmethod

from PIL import Image


class Agent(ABC):
    """
    Agent is an abstract class that defines agents used to generate the point cloud
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        next_element = self.get_next_element()

        # Check if there are no more elements to generate
        if next_element is None:
            self._i = 0
            raise StopIteration

        self._i += 1
        return next_element

    def reset_iterator(self):
        self._i = 0

    @abstractmethod
    def get_next_element(self):
        pass

    @property
    def poses(self):
        return self
