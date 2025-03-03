from __future__ import annotations

import itertools
import pdb
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import Console
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from realmdreamer.sds_dataset import SDSDataset


@dataclass
class GaussianSplattingDatamanagerConfig(VanillaDataManagerConfig):
    """Configuration for the GaussianSplattingDataManager."""

    _target: Type = field(default_factory=lambda: GaussianSplattingDatamanager)

    num_dilations: int = 1
    """Number of times to dilate the mask"""

    split_mask_by_area_threshold: bool = False
    """Whether to split the mask by area threshold"""

    area_threshold: float = 0.2
    """Area threshold for splitting the mask (percentage of image size) - [0, 1]"""

    debug: bool = False
    """Whether to only use a single image for training - for debugging"""

    select_views: bool = False
    """Whether to manually select views for training"""

    view_numbers: Optional[List[int]] = None
    """List of view numbers to use for training"""


class GaussianSplattingDatamanager(VanillaDataManager):
    """Data manager for GaussianSplatting."""

    config: GaussianSplattingDatamanagerConfig

    def create_train_dataset(self) -> SDSDataset:

        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return SDSDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            split_mask_by_area_threshold=self.config.split_mask_by_area_threshold,
            area_threshold=self.config.area_threshold,
            scale_factor=self.config.camera_res_scale_factor,
            num_dilations=self.config.num_dilations,
            single_image=self.config.debug,
        )

    def create_eval_dataset(self):
        self.eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split=self.test_split)
        return SDSDataset(
            dataparser_outputs=self.eval_dataparser_outputs,
            split_mask_by_area_threshold=self.config.split_mask_by_area_threshold,
            area_threshold=self.config.area_threshold,
            scale_factor=self.config.camera_res_scale_factor,
            num_dilations=0,
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_num_images_to_sample_from,
            num_workers=5,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        self.iter_train_image_dataloader = itertools.cycle(iter(self.train_image_dataloader))
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_cameras = self.train_dataset.cameras.to(self.device)

    def get_data_from_image_batch(self, image_batch):

        image_idx = image_batch["image_idx"].clone()

        # pdb.set_trace()
        c2w = self.train_cameras.camera_to_worlds[image_idx]

        additional = torch.tensor([0.0, 0.0, 0.0, 1.0], device=c2w.device).view(1, 1, 4).repeat(c2w.shape[0], 1, 1)

        c2w_hom = torch.cat([c2w, additional], dim=1)

        start = time.time()
        cameras = self.train_cameras[image_idx.unsqueeze(1)]
        end = time.time()

        # print(f"Time to get cameras: {end - start}")

        return cameras, image_batch.copy(), c2w_hom

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""

        if self.config.select_views:
            idx = random.choice(self.config.view_numbers)
            return self.get_i_train(idx)

        # image_batch = self.image_batch
        start = time.time()
        image_batch = next(self.iter_train_image_dataloader)
        end = time.time()

        # print(f"Time to get next batch: {end - start}")

        self.train_count += 1

        start = time.time()
        cameras, batch_all, pose_batch_c2w = self.get_data_from_image_batch(image_batch)
        end = time.time()

        # print(f"Time to get data from image batch: {end - start}")

        start = time.time()

        img_size = batch_all["image"].shape[1]

        end = time.time()

        # print(f"Time to reshape: {end - start}")

        return {
            "batch": batch_all,
            "num_images": len(image_batch["image_idx"]),
            "img_size": img_size,
            "c2w": pose_batch_c2w,
            "camera": cameras,
        }

    def get_i_train(self, idx: int) -> Tuple[RayBundle, Dict]:
        """Returns the ith data element from the train dataset"""

        image_batch = self.train_dataset[idx]
        image_batch = default_collate([image_batch])

        cameras, batch_all, pose_batch_c2w = self.get_data_from_image_batch(image_batch)

        img_size = batch_all["image"].shape[1]

        return {
            "batch": batch_all,
            "num_images": len(image_batch["image_idx"]),
            "img_size": img_size,  # Return the number of images sampled and the shape of image
            "c2w": pose_batch_c2w,
            "camera": cameras,
        }
