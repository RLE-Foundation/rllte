from typing import Callable, Dict, Tuple
import pybullet_envs
import gym as old_gym
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from hsuanwu.env.utils import HsuanwuEnvWrapper


class AdapterEnv(gym.Wrapper):
    """PyBullet robotics envs currently doesn't support Gymnasium.

    Args:
        env (Env): Environment to wrap.

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, {}

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs = self.env.reset()
        return obs, {}


def make_bullet_env(
    env_id: str = "AntBulletEnv-v0",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 0,
) -> gym.Env:
    """Build PyBullet robotics environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of parallel environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        seed (int): Random seed.

    Returns:
        Environments instance.
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
    envs = AsyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return HsuanwuEnvWrapper(envs, device)
