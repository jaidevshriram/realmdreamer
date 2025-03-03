from abc import ABC, abstractmethod

import torch

import sys

sys.path.append("/mnt/data/temp/Portal")
from utils.depth import align_depths, blend_depths


class DepthEstimator(ABC):
    """Base class for monocular depth estimation from a single image"""

    def __init__(self, cfg):

        self.cfg = cfg

    def overlap_depth_pred(
        self,
        img: torch.Tensor,  # B C H W
        depth_gt: torch.Tensor,  # B 1 H W
        mask_new: torch.Tensor,  # B 1 H W
    ):
        """
        Overlap the predicted depth with the ground truth depth
        """

        print("Base overlap depth pred")

        # valid depth mask
        mask_valid_depth = depth_gt > 0

        depth_pred = self(img)

        depth_pred_aligned = align_depths(
            source_depth=depth_pred, target_depth=depth_gt, mask=mask_valid_depth
        )

        # Blend the depths at the edges
        depth_blended = blend_depths(
            source_depth=depth_pred_aligned, target_depth=depth_gt, mask_new=mask_new
        )

        return depth_blended

    def inpaint(self, img: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor):
        raise NotImplementedError(f"Inpainting not supported for {self.__str__}")

    @staticmethod
    def align_depths(
        source_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: torch.Tensor = None,
        alignment_strategy: str = "least_squares",
    ):
        """
        Aligns the source depth with the target depth as best as possible
        """

        source_depth = (
            source_depth.cuda() if torch.cuda.is_available() else source_depth
        )
        target_depth = (
            target_depth.cuda() if torch.cuda.is_available() else target_depth
        )

        if alignment_strategy == "least_squares":

            return align_depths(source_depth=source_depth, target_depth=target_depth)

        elif alignment_strategy == "least_squares_filtered":

            aligned_depth = align_depths(
                source_depth=source_depth, target_depth=target_depth
            )

            error = torch.abs(aligned_depth - target_depth)

            mask = error < torch.quantile(error, 0.90)

            return align_depths(
                source_depth=source_depth, target_depth=target_depth, mask=mask
            )

        elif alignment_strategy == "max":

            source_depth_normalized = (source_depth - source_depth.min()) / (
                source_depth.max() - source_depth.min()
            )

            scale = (target_depth.max() - target_depth.min()) / (
                source_depth_normalized.max() - source_depth_normalized.min()
            )

            return scale * source_depth_normalized + target_depth.min()

        elif alignment_strategy == "max_filtered":

            source_depth_normalized = (source_depth - source_depth.min()) / (
                source_depth.max() - source_depth.min()
            )

            aligned_depth = align_depths(
                source_depth=source_depth_normalized, target_depth=target_depth
            )
            error = torch.abs(aligned_depth - target_depth)

            mask = error < torch.quantile(error, 0.90)

            target_depth_masked = target_depth[mask]

            scale = (target_depth_masked.max() - target_depth_masked.min()) / (
                source_depth_normalized.max() - source_depth_normalized.min()
            )

            return scale * source_depth_normalized + target_depth_masked.min()

        else:
            raise ValueError("Invalid alignment strategy" + alignment_strategy)

    @abstractmethod
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        pass
