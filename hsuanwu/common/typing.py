from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import collections
import flax
import jax

from hsuanwu.common.train_state import TrainState

PRNGKey = jax.random.PRNGKey
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, float]
TrainState = TrainState
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])