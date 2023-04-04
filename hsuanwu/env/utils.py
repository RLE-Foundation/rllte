from collections import deque
import gymnasium as gym
import numpy as np

from hsuanwu.common.typing import Tuple, Env, Ndarray, Tensor, Any, Dict


class FrameStack(gym.Wrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    Args:
        env (Env): Environment to wrap.
        k: Number of stacked frames.

    Returns:
        FrameStackEnv instance.
    """

    def __init__(self, env: Env, k: int) -> None:
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs) -> Tuple[Tensor, Dict]:
        obs, info = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: Tuple[float]) -> Tuple[Any, float, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> Ndarray:
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)