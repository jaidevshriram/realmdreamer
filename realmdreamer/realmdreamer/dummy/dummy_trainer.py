# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import pdb
import torch
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter

# from nerfstudio.viewer.server.viewer_state import ViewerState
# from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState

TRAIN_INTERATION_OUTPUT = Tuple[
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]
TORCH_DEVICE = str


@dataclass
class DummyTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: DummyTrainer)


class DummyTrainer(Trainer):

    def setup(self) -> None:
        super().setup()

        self.pipeline.config.max_num_iterations = self.config.max_num_iterations

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state="completed",
        )

    def train(self) -> None:
        """Train the model."""
        assert (
            self.pipeline.datamanager.train_dataset is not None
        ), "Missing DatsetInputs"

        torch.set_float32_matmul_precision("medium")

        self._init_viewer_state()

        print("Viewer running...")

        while True:
            pass

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()
