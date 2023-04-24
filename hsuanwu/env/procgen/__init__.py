from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from gymnasium.wrappers import NormalizeReward, RecordEpisodeStatistics, TransformObservation, TransformReward
from procgen import ProcgenEnv

from hsuanwu.env.utils import HsuanwuEnvWrapper


class AdapterEnv(gym.Wrapper):
    """Procgen games currently doesn't support Gymnasium.

    Args:
        env (Env): Environment to wrap.
        num_envs (int): Number of parallel environments.

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
        self.envs = [0] * num_envs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, {}

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs = self.env.reset()
        return obs, {}


def make_procgen_env(
    env_id: str = "bigfish",
    num_envs: int = 64,
    gamma: float = 0.99,
    num_levels: int = 0,
    start_level: int = 0,
    distribution_mode: str = "easy",
    device: str = "cpu",
) -> gym.Env:
    """Build Prcogen environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of parallel environments.
        gamma (float): A discount factor.
        num_levels (int): The number of unique levels that can be generated. Set to 0 to use unlimited levels.
        start_level (int): The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
        distribution_mode (str): What variant of the levels to use, the options are "easy", "hard", "extreme", "memory", "exploration".
        device (str): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Environments instance.
    """
    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    envs = AdapterEnv(envs, num_envs)
    envs = TransformObservation(envs, lambda obs: obs["rgb"].transpose(0, 3, 1, 2))
    envs = RecordEpisodeStatistics(envs)
    envs = NormalizeReward(envs, gamma=gamma)
    envs = TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    return HsuanwuEnvWrapper(envs, device)
