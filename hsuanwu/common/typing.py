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
from gym import Env, Space
from numpy import ndarray
from torch.distributions import Distribution
from torch.utils.data import DataLoader

Tensor = torch.Tensor
DictConfig = omegaconf.DictConfig
Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "dones", "next_observations"]
)
