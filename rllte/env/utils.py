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

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from rllte.common.preprocessing import get_preprocess_obs_fn

def make_rllte_env(
    env_id: Union[str, Callable[..., gym.Env]],
    num_envs: int = 1,
    seed: int = 1,
    device: str = "cpu",
    parallel: bool = True,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> gym.Wrapper:
    """Create environments that adapt to rllte engine.

    Args:
        env_id (Union[str, Callable[..., gym.Env]]): either the env ID, the env class or a callable returning an env
        num_envs (int): Number of environments.
        seed (int): Random seed.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        parallel (bool): `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.
        env_kwargs: Optional keyword argument to pass to the env constructor

    Returns:
        Environment wrapped by `TorchVecEnvWrapper`.
    """
    env_kwargs = env_kwargs or {}

    def make_env(rank: int) -> Callable:
        def _thunk() -> gym.Env:
            assert env_kwargs is not None
            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)
                except Exception:
                    env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)

            env.action_space.seed(seed + rank)

            return env

        return _thunk

    env_fns = [make_env(rank=i) for i in range(num_envs)]
    if parallel:
        envs = AsyncVectorEnv(env_fns)
    else:
        envs = SyncVectorEnv(env_fns)

    envs = RecordEpisodeStatistics(envs)
    # check if the environment has dictionary observation space
    if isinstance(envs.single_observation_space, gym.spaces.Dict):
        envs = TorchVecDictEnvWrapper(env=envs, device=device)
    else:
        envs = TorchVecEnvWrapper(env=envs, device=device)

    return envs


class TimeStep(NamedTuple):
    """Environment data of a time step.

    Args:
        observations (th.Tensor): Observations.
        rewards (th.Tensor): Rewards.
        actions (th.Tensor): Actions.
        terminateds (th.Tensor): Termination signal.
        truncateds (th.Tensor): Truncation signal.
        info (Dict): Extra information.
        next_observations (th.Tensor): Next observations.
    
    Returns:
        TimeStep: A time step.
    """
    # Environment data.
    observations: Optional[th.Tensor] = None
    actions: Optional[th.Tensor] = None
    rewards: Optional[th.Tensor] = None
    terminateds: Optional[th.Tensor] = None
    truncateds: Optional[th.Tensor] = None
    info: Optional[Dict] = None
    next_observations: Optional[th.Tensor] = None

    def get_episode_statistics(self) -> Tuple[List, List]:
        """Get the episode statistics.
        """
        indices = np.nonzero(self.info["episode"]["l"])
        
        return self.info["episode"]["r"][indices].tolist(), self.info["episode"]["l"][indices].tolist()

    def __getitem__(self, attr: str) -> Tuple:
        """Get the attribute of the time step.

        Args:
            attr (str): Attribute name.

        Returns:
            Tuple: Attribute value.
        """
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


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
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = env.num_envs
        self.device = th.device(device)
        # container for current observations
        self.current_obs = None
        # observation preprocessing
        self.preprocess_obs_fn = get_preprocess_obs_fn(self.observation_space)

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> TimeStep:
        """Reset all environments and return a batch of initial observations and info.

        Args:
            seed (int): The environment reset seeds.
            options (Optional[dict]): If to return the options.

        Returns:
            A `TimeStep` instance that contains first observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.preprocess_obs_fn(obs).to(self.device)
        # obs = th.as_tensor(obs, device=self.device)

        self.current_obs = obs
        return TimeStep(observations=self.current_obs, info=info)

    def step(self, actions: th.Tensor) -> TimeStep:
        """Take an action for each environment.

        Args:
            actions (th.Tensor): element of :attr:`action_space` Batch of actions.

        Returns:
            A `TimeStep` instance that contains (observation, action, reward, termination, truncations, info, next_observation).
        """
        new_obs, rewards, terminateds, truncateds, info = self.env.step(actions.cpu().numpy())
        # get real next observations
        for idx, (term, trunc) in enumerate(zip(terminateds, truncateds)):
            if term or trunc:
                new_obs[idx] = info['final_observation'][idx]

        # convert to tensor
        new_obs = self.preprocess_obs_fn(new_obs).to(self.device)
        # new_obs = th.as_tensor(new_obs, device=self.device)
        rewards = th.as_tensor(rewards, dtype=th.float32, device=self.device)

        terminateds = th.as_tensor(
            [1.0 if _ else 0.0 for _ in terminateds],
            dtype=th.float32,
            device=self.device,
        )
        truncateds = th.as_tensor(
            [1.0 if _ else 0.0 for _ in truncateds],
            dtype=th.float32,
            device=self.device,
        )

        time_step = TimeStep(observations=self.current_obs, 
                             actions=actions, 
                             rewards=rewards, 
                             terminateds=terminateds, 
                             truncateds=truncateds,
                             info=info,
                             next_observations=new_obs)

        # set current observation
        self.current_obs = new_obs

        return time_step

class TorchVecDictEnvWrapper(gym.Wrapper):
    """Env wrapper for outputting torch tensors.

    Args:
        env (VectorEnv): The vectorized environments with dictionary observations.
        device (str): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        TorchVecEnvWrapper instance.
    """

    def __init__(self, env: VectorEnv, device: str) -> None:
        super().__init__(env)
        assert isinstance(env.single_observation_space, gym.spaces.Dict), \
            f"Expected Dict observation space, got {type(env.single_observation_space)}"
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = env.num_envs
        self.device = th.device(device)
        # container for current observations
        self.current_obs = None
        # observation preprocessing
        self.preprocess_obs_fn_dict = {}

        for (key, subspace) in self.observation_space.spaces.items():
            self.preprocess_obs_fn_dict[key] = get_preprocess_obs_fn(subspace)

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ) -> TimeStep:
        """Reset all environments and return a batch of initial observations and info.

        Args:
            seed (int): The environment reset seeds.
            options (Optional[dict]): If to return the options.

        Returns:
            A `TimeStep` instance that contains first observation and info.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        for key, _obs in obs.items():
            obs[key] = self.preprocess_obs_fn_dict[key](_obs).to(self.device)

        self.current_obs = obs
        return TimeStep(observations=self.current_obs, info=info)

    def step(self, actions: th.Tensor) -> TimeStep:
        """Take an action for each environment.

        Args:
            actions (th.Tensor): element of :attr:`action_space` Batch of actions.

        Returns:
            A `TimeStep` instance that contains (observation, action, reward, termination, truncations, info, next_observation).
        """
        new_obs, rewards, terminateds, truncateds, info = self.env.step(actions.cpu().numpy())
        # get real next observations
        for idx, (term, trunc) in enumerate(zip(terminateds, truncateds)):
            if term or trunc:
                new_obs[idx] = info['final_observation'][idx]

        # convert to tensor
        for key, _obs in new_obs.items():
            new_obs[key] = self.preprocess_obs_fn_dict[key](_obs).to(self.device)
        rewards = th.as_tensor(rewards, dtype=th.float32, device=self.device)

        terminateds = th.as_tensor(
            [1.0 if _ else 0.0 for _ in terminateds],
            dtype=th.float32,
            device=self.device,
        )
        truncateds = th.as_tensor(
            [1.0 if _ else 0.0 for _ in truncateds],
            dtype=th.float32,
            device=self.device,
        )

        time_step = TimeStep(observations=self.current_obs, 
                             actions=actions, 
                             rewards=rewards, 
                             terminateds=terminateds, 
                             truncateds=truncateds,
                             info=info,
                             next_observations=new_obs)

        # set current observation
        self.current_obs = new_obs

        return time_step

class FrameStack(gym.Wrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    Args:
        env (gym.Env): Environment to wrap.
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