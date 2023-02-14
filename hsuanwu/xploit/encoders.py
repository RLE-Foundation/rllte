from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
from gym import spaces

import jax
import jax.numpy as jnp
import flax.linen as nn


class CnnEncoder(nn.Module):
    """
    Convolutional neural network for processing image-based observations.

    """

    @nn.compact
    def __call__(self, obs: jnp.array) -> jnp.array:
        x = nn.Conv(32, (3, 3), (2, 2))(obs)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)        
        return x.reshape((x.shape[0], -1))

class MlpEncoder(nn.Module):
    """
    Multi layer perceptron (MLP) for processing state-based inputs.

    """

    @nn.compact
    def __call__(self, obs: jnp.array) -> jnp.array:
        x = nn.Dense(64)(obs)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return x