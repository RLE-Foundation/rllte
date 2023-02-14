from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import flax
import jax

PRNGKey = jax.random.PRNGKey
Params = flax.core.FrozenDict[str, Any]
InfoDict = Dict[str, float]
