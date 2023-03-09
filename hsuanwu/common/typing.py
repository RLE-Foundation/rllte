from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union, Iterable
from torch.distributions import Distribution
from torch.utils.data import DataLoader
from gym import Space, Env
from numpy import ndarray
from pathlib import Path
import omegaconf
import collections
import torch

Tensor = torch.Tensor
DictConfig = omegaconf.DictConfig
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'dones', 'next_observations'])