from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics


class VecEnvWrapper(gym.Wrapper):
    """Env wrapper for adapting to rllte engine and outputting torch tensors.

    Args:
        env_id (Union[str, Callable[..., gym.Env]]): either the env ID, the env class or a callable returning an env
        num_envs (int): Number of environments.
        seed (int): Random seed.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        parallel (bool): `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.
        env_kwargs: Optional keyword argument to pass to the env constructor

    Returns:
        VecEnvWrapper instance.
    """

    def __init__(
        self,
        env_id: Union[str, Callable[..., gym.Env]],
        num_envs: int = 1,
        seed: int = 1,
        device: str = "cpu",
        parallel: bool = True,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        env_kwargs = env_kwargs or {}

        def make_env(rank: int) -> Callable:
            def _thunk() -> gym.Env:
                assert env_kwargs is not None
                if isinstance(env_id, str):
                    # if the render mode was not specified, we set it to `rgb_array` as default.
                    kwargs = {"render_mode": "rgb_array"}
                    kwargs.update(env_kwargs)
                    try:
                        env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                    except Exception:
                        env = gym.make(env_id, **env_kwargs)
                else:
                    env = env_id(**env_kwargs)

                env.action_space.seed(seed + rank)

                return env

            return _thunk

        env_fns = [make_env(rank=i) for i in range(num_envs)]
        if parallel:
            env = AsyncVectorEnv(env_fns)
        else:
            env = SyncVectorEnv(env_fns)

        env = RecordEpisodeStatistics(env)
        env = TorchVecEnvWrapper(env=env, device=device)
        super().__init__(env)


class TorchVecEnvWrapper(gym.Wrapper):
    """Env wrapper for outputting torch tensors.

    Args:
        env (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        TorchVecEnvWrapper instance.
    """

    def __init__(self, env: VectorEnv, device: str) -> None:
        super().__init__(env)
        self.device = th.device(device)

        # TODO: Transform the original 'Box' space into Hydra supported type.
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = env.num_envs

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> Tuple[th.Tensor, Dict]:
        """Reset all environments and return a batch of initial observations and info.

        Args:
            seed (int): The environment reset seeds.
            options (Optional[dict]): If to return the options.

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        obs = th.as_tensor(obs, device=self.device)
        return obs, info

    def step(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, bool, Dict]:
        """Take an action for each environment.

        Args:
            actions (Tensor): element of :attr:`action_space` Batch of actions.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos)
        """
        obs, reward, terminated, truncated, info = self.env.step(actions.cpu().numpy())
        obs = th.as_tensor(obs, device=self.device)
        reward = th.as_tensor(reward, dtype=th.float32, device=self.device)
        terminated = th.as_tensor(
            [1.0 if _ else 0.0 for _ in terminated],
            dtype=th.float32,
            device=self.device,
        )
        truncated = th.as_tensor(
            [1.0 if _ else 0.0 for _ in truncated],
            dtype=th.float32,
            device=self.device,
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
