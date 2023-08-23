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


from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from gymnasium.spaces import Box
from gymnasium.wrappers import NormalizeReward, RecordEpisodeStatistics, TransformObservation, TransformReward
from procgen import ProcgenEnv

from rllte.env.utils import (Gymnasium2Torch,
                             EnvPoolAsync2Gymnasium,
                             EnvPoolSync2Gymnasium)

class AdapterEnv(gym.Wrapper):
    """Procgen games currently doesn't support Gymnasium.

    Args:
        env (gym.Env): Environment to wrap.
        num_envs (int): Number of environments.

    Returns:
        AdapterEnv instance.
    """

    def __init__(self, env: gym.Env, num_envs: int) -> None:
        super().__init__(env)
        self.single_observation_space = Box(
            low=env.observation_space["rgb"].low[0, 0, 0],
            high=env.observation_space["rgb"].high[0, 0, 0],
            shape=[3, 64, 64],
            dtype=env.observation_space["rgb"].dtype,
        )
        self.single_action_space = env.action_space
        self.is_vector_env = True
        self.num_envs = num_envs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment.

        Args:
            action (int): Action to take.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, {}

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs = self.env.reset()
        return obs, {}
    

def make_envpool_procgen_env(
    env_id: str = "bigfish",
    num_envs: int = 64,
    device: str = "cpu",
    seed: int = 1,
    gamma: float = 0.99,
    num_levels: int = 200,
    start_level: int = 0,
    distribution_mode: str = "easy",
    parallel: bool = True,
) -> gym.Env:
    """Create Procgen environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        gamma (float): A discount factor.
        num_levels (int): The number of unique levels that can be generated.
            Set to 0 to use unlimited levels.
        start_level (int): The lowest seed that will be used to generated levels.
            'start_level' and 'num_levels' fully specify the set of possible levels.
        distribution_mode (str): What variant of the levels to use, the options are "easy",
            "hard", "extreme", "memory", "exploration".
        parallel (bool): `True` for creating asynchronous environments, and `False`
            for creating synchronous environments.
            
    Returns:
        The vectorized environments.
    """
    if distribution_mode == "easy":
        task_id = env_id.capitalize() + "Easy-v0"
    elif distribution_mode == "hard":
        task_id = env_id.capitalize() + "Hard-v0"
    else:
        raise NotImplementedError(f"Distribution mode `{distribution_mode}` is not implemented in `EnvPool`!")

    env_kwargs = dict(
        task_id=task_id,
        env_type="gymnasium",
        num_envs=num_envs,
        batch_size=num_envs,
        seed=seed,
        env_name=env_id,
        num_levels=num_levels,
        start_level=start_level,
    )
    if parallel:
        envs = EnvPoolAsync2Gymnasium(env_kwargs)
    else:
        envs = EnvPoolSync2Gymnasium(env_kwargs)

    envs = RecordEpisodeStatistics(envs)
    envs = NormalizeReward(envs, gamma=gamma)
    envs = TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    return Gymnasium2Torch(envs, device=device, envpool=True)


def make_procgen_env(
    env_id: str = "bigfish",
    num_envs: int = 64,
    device: str = "cpu",
    seed: int = 1,
    gamma: float = 0.99,
    num_levels: int = 200,
    start_level: int = 0,
    distribution_mode: str = "easy",
) -> gym.Env:
    """Create Procgen environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        gamma (float): A discount factor.
        num_levels (int): The number of unique levels that can be generated.
            Set to 0 to use unlimited levels.
        start_level (int): The lowest seed that will be used to generated levels.
            'start_level' and 'num_levels' fully specify the set of possible levels.
        distribution_mode (str): What variant of the levels to use, the options are "easy",
            "hard", "extreme", "memory", "exploration".

    Returns:
        The vectorized environment.
    """
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        rand_seed=seed,
    )
    envs = AdapterEnv(envs, num_envs)
    envs = TransformObservation(envs, lambda obs: obs["rgb"].transpose(0, 3, 1, 2))
    envs = RecordEpisodeStatistics(envs)
    envs = NormalizeReward(envs, gamma=gamma)
    envs = TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    return Gymnasium2Torch(envs, device)
