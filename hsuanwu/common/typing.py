from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from torch.distributions import Distribution
from gym import Space, Env
from numpy import ndarray
import collections
import torch

Tensor = torch.Tensor
InfoDict = Dict[str, float]
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'dones', 'next_observations'])