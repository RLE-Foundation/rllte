# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

from typing import Callable, Tuple, Dict, Union
from gymnasium import spaces

import gymnasium as gym
import numpy as np


def process_observation_space(observation_space: gym.Space) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """Process the observation space.
    
    Args:
        observation_space (gym.Space): Observation space.

    Returns:
        Information of the observation space.
    """
    if isinstance(observation_space, spaces.Box):
        # Observation is a vector
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: process_observation_space(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
    

def process_action_space(action_space: gym.Space) -> Tuple[int, str, Union[int, float]]:
    """Get the dimension of the action space.

    Args:
        action_space (gym.Space): Action space.

    Returns:
        Information of the action space.
    """
    # TODO: revise the action_range
    action_shape = action_space.shape
    if action_space.__class__.__name__ == "Discrete":
        action_dim = int(action_space.n)
        action_type = "Discrete"
        action_range = [0, int(action_space.n) - 1]
    elif action_space.__class__.__name__ == "Box":
        action_dim = int(np.prod(action_space.shape))
        action_type = "Box"
        action_range = [
            float(action_space.low[0]),
            float(action_space.high[0]),
        ]
    elif action_space.__class__.__name__ == "MultiDiscrete":
        action_dim = int(len(action_space.nvec))
        action_type = "MultiDiscrete"
        action_range = [0, int(action_space.nvec[0]) - 1]
    elif action_space.__class__.__name__ == "MultiBinary":
        action_dim = int(action_space.shape[0])
        action_type = "MultiBinary"
        action_range = [0, 1]
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
    
    return action_shape, action_dim, action_type, action_range
    

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """Get the dimension of the observation space when flattened. It does not apply to image observation space.
        Borrowed from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py#L169

    Args:
        observation_space (spaces.Space): Observation space.

    Returns:
        The dimension of the observation space when flattened.
    """
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


def process_env_info(observation_space: gym.Space, action_space: gym.Space) -> Tuple[Tuple, ...]:
    """Process the environment information.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.

    Returns:
        Information of the observation and action space.
    """
    # observation part
    obs_shape = process_observation_space(observation_space)
    # action part
    action_shape, action_dim, action_type, action_range = process_action_space(action_space)

    return obs_shape, action_shape, action_dim, action_type, action_range