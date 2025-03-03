from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

import pdb
import time
import itertools
import torch
from torch.utils.data import DataLoader
from nerfstudio.utils.rich_utils import CONSOLE

from .dummy_dataset import DummyDataset


@dataclass
class DummyDatamanagerConfig(VanillaDataManagerConfig):
    """Configuration for the GaussianSplattingDataManager."""

    _target: Type = field(default_factory=lambda: DummyDatamanager)


class DummyDatamanager(VanillaDataManager):
    """Data manager for GaussianSplatting."""

    config: DummyDatamanagerConfig

    def __init__(
        self,
        config: DummyDatamanagerConfig,
        device: str = "cpu",
        test_mode: str = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.includes_time = False
        pass

    def create_train_dataset(self):
        return DummyDataset(self.train_dataparser_outputs, 1)

    def create_eval_dataset(self):
        return DummyDataset(self.train_dataparser_outputs, 1)

    def setup_train(self):
        pass

    def get_i_train(self, idx: int) -> Tuple[RayBundle, Dict]:
        return {}
