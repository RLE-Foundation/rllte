from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector.vector_env import VectorEnv
from omegaconf import OmegaConf


class HsuanwuEnvWrapper(gym.Wrapper):
    """Env wrapper for adapting to Hsuanwu engine and outputting torch tensors.

    Args:
        env (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        HsuanwuEnvWrapper instance.
    """

    def __init__(self, env: VectorEnv, device: str) -> None:
        super().__init__(env)
        self._device = th.device(device)

        # TODO: Transform the original 'Box' space into Hydra supported type.
        self.observation_space = OmegaConf.create({"shape": env.single_observation_space.shape})

        if env.single_action_space.__class__.__name__ == "Discrete":
            n = int(env.single_action_space.n)
            self.action_space = OmegaConf.create({"shape": (n,), "type": "Discrete", "range": [0, n - 1]})
        elif env.single_action_space.__class__.__name__ == "Box":
            low, high = float(env.single_action_space.low[0]), float(env.single_action_space.high[0])
            self.action_space = OmegaConf.create(
                {
                    "shape": env.single_action_space.shape,
                    "type": "Box",
                    "range": [low, high],
                }
            )
        else:
            raise NotImplementedError("Unsupported action type!")
        self.num_envs = len(env.envs)

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> Tuple[th.Tensor, Dict]:
        """Reset all parallel environments and return a batch of initial observations and info.

        Args:
            seed (int): The environment reset seeds.
            options (Optional[dict]): If to return the options.

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        obs = th.as_tensor(obs, device=self._device)
        return obs, info

    def step(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, bool, Dict]:
        """Take an action for each parallel environment.

        Args:
            actions (Tensor): element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        if self.action_space["type"] == "Discrete":
            actions = actions.squeeze(1).cpu().numpy()
        else:
            actions = actions.cpu().numpy()

        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = th.as_tensor(obs, device=self._device)
        reward = th.as_tensor(reward, dtype=th.float32, device=self._device)
        terminated = th.as_tensor(
            [1.0 if _ else 0.0 for _ in terminated],
            dtype=th.float32,
            device=self._device,
        )
        truncated = th.as_tensor(
            [1.0 if _ else 0.0 for _ in truncated],
            dtype=th.float32,
            device=self._device,
        )

        return obs, reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    Args:
        env (Env): Environment to wrap.
        k (int): Number of stacked frames.

    Returns:
        FrameStackEnv instance.
    """

    def __init__(self, env: gym.Env, k: int) -> None:
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

    def reset(self, **kwargs) -> Tuple[th.Tensor, Dict]:
        obs, info = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: Tuple[float]) -> Tuple[Any, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
