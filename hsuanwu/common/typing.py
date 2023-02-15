from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from gym import Space, Env
import collections
import torch

Tensor = torch.Tensor
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])