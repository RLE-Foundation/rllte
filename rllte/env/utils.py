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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import envpool
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics


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
        device (str): Device to convert data.
        parallel (bool): `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.
        env_kwargs: Optional keyword argument to pass to the env constructor.

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

    return Gymnasium2Torch(env=envs, device=device)


class EnvPoolAsync2Gymnasium(gym.Wrapper):
    """Create an `EnvPool` environment with asynchronous mode, and wrap it 
        to allow a modular transformation of the `step` and `reset` methods.
    
    Args:
        env_kwargs (gym.Env): Environment arguments.
    
    Returns:
        A `Gymnasium`-like environment.
    """
    def __init__(self, env_kwargs: Dict) -> None:
        env = envpool.make(**env_kwargs)
        super().__init__(env)
        self.num_envs = env_kwargs.get("num_envs", 1)
        self.is_vector_env = True
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        # send the initial reset signal to all envs
        self.env.async_reset()
        obs, rew, term, trunc, info = self.env.recv()
        # run one step to get the initial observation
        self.env.send(np.zeros(shape=(self.num_envs, *self.action_space.shape)), info["env_id"])
        return obs, info

    def step(self, actions: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Step the environment.

        Args:
            actions (int): Action to take.

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        obs, rew, term, trunc, info = self.env.recv()
        self.env.send(actions, info["env_id"])

        return obs, rew, term, trunc, info


class EnvPoolSync2Gymnasium(gym.Wrapper):
    """Wraps an `EnvPool` environment with synchronous mode to allow 
        a modular transformation of the `step` and `reset` methods.

    Args:
        env_kwargs (gym.Env): Environment arguments.

    Returns:
        A `Gymnasium`-like environment.
    """
    def __init__(self, env_kwargs: Dict) -> None:
        env = envpool.make(**env_kwargs)
        super().__init__(env)
        self.num_envs = env_kwargs.get("num_envs", 1)
        self.is_vector_env = True

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment with `envpool`."""
        return self.env.reset()

    def step(self, actions: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Step the environment with `envpool`.

        Args:
            actions (int): Action to take.

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        return self.env.step(actions)


class Gymnasium2Torch(gym.Wrapper):
    """Env wrapper for processing gymnasium environments and outputting torch tensors.

    Args:
        env (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        envpool (bool): Whether to use `EnvPool` env.

    Returns:
        Gymnasium2Torch wrapper.
    """

    def __init__(self, env: VectorEnv, device: str, envpool: bool = False) -> None:
        super().__init__(env)
        self.num_envs = env.num_envs
        self.device = th.device(device)

        # envpool's observation space and action space are the same as the single env.
        if not envpool:
            self.observation_space = env.single_observation_space
            self.action_space = env.single_action_space

        if isinstance(self.observation_space, gym.spaces.Dict):
            self._format_obs = lambda x: {key: th.as_tensor(item, device=self.device) for key, item in x.items()}
        else:
            self._format_obs = lambda x: th.as_tensor(x, device=self.device)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[th.Tensor, Dict]:
        """Reset all environments and return a batch of initial observations and info.

        Args:
            seed (int): The environment reset seeds.
            options (Optional[dict]): If to return the options.

        Returns:
            First observations and info.
        """
        obs, infos = self.env.reset(seed=seed, options=options)

        return self._format_obs(obs), infos

    def step(self, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, List[Dict]]:
        """Take an action for each environment.

        Args:
            actions (th.Tensor): element of :attr:`action_space` Batch of actions.

        Returns:
            Next observations, rewards, terminateds, truncateds, infos.
        """
        new_observations, rewards, terminateds, truncateds, infos = self.env.step(actions.cpu().numpy())
        # TODO: get real next observations
        # for idx, (term, trunc) in enumerate(zip(terminateds, truncateds)):
        #     if term or trunc:
        #         new_obs[idx] = info['final_observation'][idx]

        # convert to tensor
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

        return self._format_obs(new_observations), rewards, terminateds, truncateds, infos


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


class DistributedWrapper:
    """An env wrapper to adapt to the distributed trainer.

    Args:
        env (gym.Env): A Gym-like env.

    Returns:
        Processed env.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.episode_return = None
        self.episode_step = None
        if env.action_space.__class__.__name__ == "Discrete":
            self.action_type = "Discrete"
            self.action_dim = 1
        elif env.action_space.__class__.__name__ == "Box":
            self.action_type = "Box"
            self.action_dim = env.action_space.shape[0]
        else:
            raise NotImplementedError("Unsupported action type!")

    def reset(self, seed) -> Dict[str, th.Tensor]:
        """Reset the environment."""
        init_reward = th.zeros(1, 1)
        init_last_action = th.zeros(1, self.action_dim, dtype=th.int64)
        self.episode_return = th.zeros(1, 1)
        self.episode_step = th.zeros(1, 1, dtype=th.int32)
        init_terminated = th.ones(1, 1, dtype=th.uint8)
        init_truncated = th.ones(1, 1, dtype=th.uint8)

        obs, info = self.env.reset(seed=seed)
        obs = self._format_obs(obs)

        return dict(
            observations=obs,
            rewards=init_reward,
            terminateds=init_terminated,
            truncateds=init_truncated,
            episode_returns=self.episode_return,
            episode_steps=self.episode_step,
            last_actions=init_last_action,
        )

    def step(self, action: th.Tensor) -> Dict[str, th.Tensor]:
        """Step function that returns a dict consists of the current and history observation and action.

        Args:
            action (th.Tensor): Action tensor.

        Returns:
            Step information dict.
        """
        if self.action_type == "Discrete":
            _action = action.item()
        elif self.action_type == "Box":
            _action = action.squeeze(0).cpu().numpy()
        else:
            raise NotImplementedError("Unsupported action type!")

        obs, reward, terminated, truncated, info = self.env.step(_action)
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if terminated or truncated:
            obs, info = self.env.reset()
            self.episode_return = th.zeros(1, 1)
            self.episode_step = th.zeros(1, 1, dtype=th.int32)

        obs = self._format_obs(obs)
        reward = th.as_tensor(reward, dtype=th.float32).view(1, 1)
        terminated = th.as_tensor(terminated, dtype=th.uint8).view(1, 1)
        truncated = th.as_tensor(truncated, dtype=th.uint8).view(1, 1)

        return dict(
            observations=obs,
            rewards=reward,
            terminateds=terminated,
            truncateds=truncated,
            episode_returns=episode_return,
            episode_steps=episode_step,
            last_actions=action
        )

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def _format_obs(self, obs: np.ndarray) -> th.Tensor:
        """Reformat the observation by adding an time dimension.

        Args:
            obs (np.ndarray): Observation.

        Returns:
            Formatted observation.
        """
        obs = th.from_numpy(np.array(obs))
        return obs.view((1, 1, *obs.shape))
