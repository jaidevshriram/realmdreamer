import cv2
import os
import json

from tqdm.auto import tqdm

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipelineConfig

import pdb
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import kornia
import matplotlib.pyplot as plt
import torch
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.nn import Parameter
from torchvision import transforms
from typing_extensions import Literal

import wandb
from .dummy_datamanager import DummyDatamanagerConfig

import time


@dataclass
class DummyPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DummyPipeline)
    """target class to instantiate"""

    datamanager: DummyDatamanagerConfig = DummyDatamanagerConfig()
    """specifies the datamanager config"""


class DummyPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: DummyPipelineConfig

    def __init__(
        self,
        config: DummyPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        grad_scaler=None,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.datamanager = config.datamanager._target(
            config.datamanager, device, test_mode, world_size, local_rank
        )
        pass

    def get_param_groups(self):
        return {}

    def get_eval_loss_dict(self, step: int):
        return {}

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
