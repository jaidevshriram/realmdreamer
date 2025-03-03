import pdb
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import kornia
import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image


def cleanup_mask(mask, num_dilations):
    mask = 1 - mask.float()

    mask = mask.unsqueeze(0).unsqueeze(0)

    mask = kornia.morphology.opening(mask.float(), kernel=torch.ones(5, 5).to(mask.device))

    for _ in range(num_dilations):
        mask = kornia.morphology.dilation(mask, kernel=torch.ones(5, 5).to(mask.device))

    return mask.squeeze(0).squeeze(0)


# input_image - PIL image, area_threshold - int
def split_image_by_area_threshold(input_image, area_threshold):
    # Convert the PIL image to a NumPy array
    image_array = np.array(input_image)

    # Convert to binary image (assuming it's not already binary)
    binary_image = np.where(image_array > 0, 0, 1)

    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image.astype(np.uint8))

    # Create empty arrays to store separated regions
    region_greater_than_threshold = np.ones_like(binary_image)
    region_less_than_threshold = np.ones_like(binary_image)

    # Iterate over labeled components
    for label in range(1, num_labels):  # Skip background label 0
        area = stats[label, cv2.CC_STAT_AREA]

        if area > area_threshold:
            region_greater_than_threshold[labels == label] = 0
        else:
            region_less_than_threshold[labels == label] = 0

    # Convert back to PIL images
    image_greater_than_threshold = Image.fromarray((region_greater_than_threshold * 255).astype(np.uint8))
    image_less_than_threshold = Image.fromarray((region_less_than_threshold * 255).astype(np.uint8))

    return image_greater_than_threshold, image_less_than_threshold


class SDSDataset(DepthDataset):
    """Dataset that returns images and inpainting masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        split_mask_by_area_threshold: bool = False,
        area_threshold: float = 0.1,
        scale_factor: float = 1.0,
        num_dilations=0,
        single_image=False,
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert "mask_inpainting_filenames" in dataparser_outputs.metadata.keys()
        # assert "rendered_rgb_filenames" in dataparser_outputs.metadata.keys()
        self.inpainting_masks = self.metadata["mask_inpainting_filenames"]

        if self.inpainting_masks is None:
            raise ValueError("Inpainting masks are required for this dataset - maybe the dataset is too small, so the split for this is not available")

        # self.rendered_rgb_filenames = self.metadata["rendered_rgb_filenames"]
        self.num_dilations = num_dilations
        self.cache = {}
        self.cached = [False for _ in range(len(self.inpainting_masks))]
        self.split_mask_by_area_threshold = split_mask_by_area_threshold
        self.area_threshold = area_threshold
        self.single_image = single_image

    def get_metadata(self, data: Dict) -> Dict:

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        newsize = (int(width * self.scale_factor), int(height * self.scale_factor))

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale

        depth_image = get_depth_image_from_path(
            filepath=filepath,
            height=newsize[1],
            width=newsize[0],
            scale_factor=scale_factor,
        )

        metadata = {"depth_image": depth_image}

        # handle mask
        mask_filepath = self.inpainting_masks[data["image_idx"]]

        # Inpainting mask
        mask = Image.open(mask_filepath)

        if self.split_mask_by_area_threshold:

            if self.scale_factor != 1.0:
                width, height = mask.size
                newsize = (
                    int(width * self.scale_factor),
                    int(height * self.scale_factor),
                )

            # Split the mask into two masks based on area threshold
            mask_greater_than_threshold, mask_less_than_threshold = split_image_by_area_threshold(
                mask, self.area_threshold * width * height
            )

            # Resize the masks
            mask_greater_than_threshold = mask_greater_than_threshold.resize(newsize, resample=Image.ANTIALIAS)
            mask_less_than_threshold = mask_less_than_threshold.resize(newsize, resample=Image.ANTIALIAS)

            # Convert to tensors
            mask_greater_than_threshold = (torch.from_numpy(np.array(mask_greater_than_threshold)) / 255.0).bool()
            mask_less_than_threshold = (torch.from_numpy(np.array(mask_less_than_threshold)) / 255.0).bool()

            # Cleanup the masks
            mask_greater_than_threshold = cleanup_mask(mask_greater_than_threshold, self.num_dilations)
            mask_less_than_threshold = cleanup_mask(mask_less_than_threshold, self.num_dilations)

            metadata.update(
                {
                    "inpainting_mask": mask_greater_than_threshold.bool(),
                    "inpainting_mask_2": mask_less_than_threshold.bool(),
                }
            )

        else:
            if self.scale_factor != 1.0:
                width, height = mask.size
                newsize = (
                    int(width * self.scale_factor),
                    int(height * self.scale_factor),
                )
                mask = mask.resize(newsize, resample=Image.ANTIALIAS)

            mask = (torch.from_numpy(np.array(mask)) / 255.0).bool()  # Shape is (H, W)
            mask = cleanup_mask(mask, self.num_dilations)

            metadata.update(
                {
                    "inpainting_mask": mask.bool(),
                    "inpainting_mask_2": mask.bool(),  # This makes mask 2 the same as mask 1
                }
            )

        return metadata

    def __getitem__(self, image_idx: int) -> Dict:

        if self.single_image:
            image_idx = 0
            if len(self.cached) > 45:
                image_idx = 45

        if self.cached[image_idx]:
            data = self.cache[image_idx]
        else:
            data = self.get_data(image_idx)
            self.cache[image_idx] = data
            self.cached[image_idx] = True

        data["image_idx"] = image_idx

        return data


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        # image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        CONSOLE.print("[bold red]Only .npy files should be supported for depth - PNG might have bugs")

        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)

        image = image.astype(np.float64)

        # -1 gets mapped to 64536 in the depth image, set to -1000
        image[image == 64536] = -1000

        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])
