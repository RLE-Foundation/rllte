import collections
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import omegaconf
import torch
import numpy as np
from gymnasium import Env, Space
from torch.distributions import Distribution
from torch.utils.data import DataLoader

Storage = ''
TorchSize = torch.Size()
NNModule = torch.nn.Module
Ndarray = np.ndarray
Tensor = torch.Tensor
Device = torch.device
DictConfig = omegaconf.DictConfig
Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "dones", "next_observations"]
)
