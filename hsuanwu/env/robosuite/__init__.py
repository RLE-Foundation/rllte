from typing import Callable, Dict, List, Any, Tuple
import gymnasium as gym
import numpy as np
from collections import deque
import torch as th
from hsuanwu.env.utils import TorchVecEnvWrapper
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

class PixelEnv(gym.Env):
    """Observation wrapper that stacks the observations in a rolling manner.

    Args:
        env (Env): Environment to wrap.
        k (int): Number of stacked frames.

    Returns:
        FrameStackEnv instance.
    """

    def __init__(self) -> None:
        self._k = 3
        self._frames = deque([], maxlen=3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(9, 84, 84),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, ),
            dtype=np.float32
        )
        self.total_step = 0

    def reset(self, **kwargs) -> Tuple[th.Tensor, Dict]:
        self.total_step = 0
        obs = self.observation_space.sample()[:3,:,:]
        info = {'discount': 1.0}

        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: Tuple[float]) -> Tuple[Any, float, bool, bool, Dict]:
        # obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.observation_space.sample()[:3,:,:]
        reward = np.random.rand(1)
        if self.total_step < 499:
            terminated = truncated = False
        else:
            terminated = truncated = True
        info = {'discount': 1.0}
        self.total_step += 1

        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
    

def make_robosuite_env(
        env_id: str = "cartpole_balance",
        num_envs: int = 1,
        device: str = "cpu"):
    def make_env():
        def _thunk():
            return PixelEnv()

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device)