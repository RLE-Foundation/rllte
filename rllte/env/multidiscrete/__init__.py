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


from typing import Any, Callable, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.utils import Gymnasium2Torch


class StateEnv(gym.Env):
    """Environment with state-based observation space and `MultiDiscrete` action space for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(nvec=(2, 3, 4))

    def reset(self, seed: Optional[int] = None, options=Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int, optional): Seed for the environment. Defaults to None.
            options (Dict[str, Any], optional): Options for the environment. Defaults to None.

        Returns:
            Observation and info.
        """
        return self.observation_space.sample(), {}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action (Any): Action to take.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs = self.observation_space.sample()
        reward = 0.5
        if np.random.rand() > 0.5:
            terminated = True
        else:
            terminated = False
        truncated = terminated
        info = {}

        return obs, reward, terminated, truncated, info


class PixelEnv(gym.Env):
    """Environment with image-based observation space and `MultiDiscrete` action space for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4, 84, 84), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(nvec=(2, 3, 4))

    def reset(self, seed: Optional[int] = None, options=Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int, optional): Seed for the environment. Defaults to None.
            options (Dict[str, Any], optional): Options for the environment. Defaults to None.

        Returns:
            Observation and info.
        """
        return self.observation_space.sample(), {}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action (Any): Action to take.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs = self.observation_space.sample()
        reward = 0.5
        if np.random.rand() > 0.5:
            terminated = True
        else:
            terminated = False
        truncated = terminated
        info = {}

        return obs, reward, terminated, truncated, info


def make_multidiscrete_env(
    env_id: str = "MultiDiscrete-State", num_envs: int = 1, device: str = "cpu", seed: int = 0, parallel: bool = True
) -> gym.Env:
    """Build environments with `MultiDiscrete` action space for testing.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        parallel (bool): `True` for creating asynchronous environments, and `False`
            for creating synchronous environments.

    Returns:
        The vectorized environments.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            if env_id == "MultiDiscrete-State":
                env = StateEnv()
            else:
                env = PixelEnv()
            env.observation_space.seed(seed)
            env.action_space.seed(seed)
            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]

    if parallel:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device=device)
