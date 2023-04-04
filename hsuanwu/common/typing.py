import collections
from abc import ABC, abstractmethod
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

import numpy as np
import omegaconf
import torch
from gymnasium import Env, Space
from torch.distributions import Distribution
from torch.utils.data import DataLoader

Logger = ""
SimpleQueue = torch.multiprocessing.SimpleQueue
Storage = ""
TorchSize = torch.Size
NNModule = torch.nn.Module
Ndarray = np.ndarray
Tensor = torch.Tensor
Device = torch.device
DictConfig = omegaconf.DictConfig
Batch = collections.namedtuple(
    "Batch", ["observations", "actions", "rewards", "dones", "next_observations"]
)
