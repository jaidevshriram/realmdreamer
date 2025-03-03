from abc import ABC, abstractmethod
import numpy as np
import torch

from typing import Tuple


class PointCloudRenderer(ABC):
    """
    PointCloudRenderer is an abstract class that defines point cloud renderers
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def init_pcd_from_rgbd(
        self,
        rgb: torch.Tensor,  # B C H W,
        depth: torch.Tensor,  # B 1 H W,
        pose: np.ndarray,  # 4x4 matrix
    ):
        pass

    @abstractmethod
    def init_shadow_volume(self, pose: np.ndarray):  # 4x4 matrix
        """Create a shadow volume from the current view point"""
        pass

    @abstractmethod
    def update_pcd(
        self,
        rgb: torch.Tensor,  # B C H W,
        depth: torch.Tensor,  # B 1 H W,
        mask: torch.Tensor,  # B 1 H W,
        pose: np.ndarray,  # 4x4 matrix
    ) -> None:
        """Converts the image, depth and mask to a point cloud and updates the existing point cloud."""
        pass

    @abstractmethod
    def __call__(
        self, pose: np.ndarray, with_shadow_volume=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # pose is a 4x4 matrix
        """

        Renders the point cloud from the current pose - fixed intrinsics

        Returns:
        - rgb: B C H W
        - depth: B 1 H W
        - ids: B 1 H W

        """
        pass

    @abstractmethod
    def delete_points(self, ids):
        """Deletes points from the point cloud - ids is a tensor of shape (N, 1)"""
        pass

    @staticmethod
    @abstractmethod
    def combine_pcds(a, b):
        """Combine two instance of a point cloud"""
        pass

    @abstractmethod
    def write_to_ply(self, ply_path):
        pass
