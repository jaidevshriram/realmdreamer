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
from PIL import Image

from nerfstudio.utils.rich_utils import CONSOLE


class DummyDataset(InputDataset):
    """Dummy dataset that returns empty dict"""

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
    ):
        super().__init__(dataparser_outputs, scale_factor)

    def __getitem__(self, image_idx: int) -> Dict:
        return {}
