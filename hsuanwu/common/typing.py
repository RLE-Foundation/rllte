from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import collections
import flax
import jax

PRNGKey = jax.random.PRNGKey
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, float]
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])