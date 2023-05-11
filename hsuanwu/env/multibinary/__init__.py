from typing import Any, Callable, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from hsuanwu.env.utils import TorchVecEnvWrapper


class StateEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(n=3)

    def reset(self, seed: Optional[int] = None, options=Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        return self.observation_space.sample(), {}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
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
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4, 84, 84), dtype=np.float32)
        self.action_space = gym.spaces.MultiBinary(n=3)

    def reset(self, seed: Optional[int] = None, options=Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        return self.observation_space.sample(), {}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs = self.observation_space.sample()
        reward = 0.5
        if np.random.rand() > 0.5:
            terminated = True
        else:
            terminated = False
        truncated = terminated
        info = {}

        return obs, reward, terminated, truncated, info


def make_multibinary_env(
    env_id: str = "multibinary_state",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 0,
) -> gym.Env:
    """Build environments with `MultiBinary` action space for testing.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        seed (int): Random seed.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            if env_id == "multibinary_state":
                env = StateEnv()
            else:
                env = PixelEnv()
            env.observation_space.seed(seed)
            env.action_space.seed(seed)
            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = AsyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device)
