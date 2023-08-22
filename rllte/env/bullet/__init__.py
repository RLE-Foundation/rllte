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


from typing import Callable, Dict, Tuple

import gym as old_gym
import gymnasium as gym
import numpy as np
import pybullet_envs as pybullet_envs
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.utils import Gymnasium2Torch


class AdapterEnv(gym.Wrapper):
    """PyBullet robotics envs currently doesn't support Gymnasium.

    Args:
        env (gym.Env): Environment to wrap.

    Returns:
        AdapterEnv instance.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
            low=env.observation_space.low,
            high=env.observation_space.high,
        )
        self.action_space = gym.spaces.Box(
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
            low=env.action_space.low,
            high=env.action_space.high,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, {}

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs = self.env.reset()
        return obs, {}


def make_bullet_env(
    env_id: str = "AntBulletEnv-v0", 
    num_envs: int = 1, 
    device: str = "cpu", 
    seed: int = 0, 
    parallel: bool = True
) -> gym.Env:
    """Create PyBullet robotics environments.

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
            env = old_gym.make(env_id)
            env.seed(seed)
            env.observation_space.seed(seed)
            env.action_space.seed(seed)
            return AdapterEnv(env)

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]

    if parallel:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device=device)