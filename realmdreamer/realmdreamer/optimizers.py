from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from nerfstudio.engine.optimizers import OptimizerConfig


@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.AdamW
    weight_decay: float = 0.01
    """The weight decay to use."""
