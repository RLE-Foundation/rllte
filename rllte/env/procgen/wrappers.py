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


from typing import Any, Dict, Tuple
from gymnasium.spaces.box import Box

import envpool
import gymnasium as gym
import numpy as np

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

class EnvPoolAsynchronous(gym.Wrapper):
    """Build the environment with `envpool` and asynchronous mode.

    Args:
        env_kwargs (Dict): Environment kwargs.

    Returns:
        EnvPoolAsynchronous instance.
    """

    def __init__(self, env_kwargs: Dict) -> None:
        env = envpool.make(**env_kwargs)
        super().__init__(env)
        self.num_envs = env_kwargs["num_envs"]
        self.is_vector_env = True

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment with `envpool`."""
        # send the initial reset signal to all envs
        self.env.async_reset()
        obs, rew, term, trunc, info = self.env.recv()
        # run one step to get the initial observation
        self.env.send(np.zeros(shape=(self.num_envs,)), info["env_id"])
        return obs, info

    def step(self, actions: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Step the environment with `envpool`.

        Args:
            actions (int): Action to take.

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        obs, rew, term, trunc, info = self.env.recv()
        self.env.send(actions, info["env_id"])

        return obs, rew, term, trunc, info


class EnvPoolSynchronous(gym.Wrapper):
    """Build the environment with `envpool` and synchronous mode.

    Args:
        env_kwargs (Dict): Environment kwargs.

    Returns:
        EnvPoolAsynchronous instance.
    """

    def __init__(self,  env_kwargs: Dict) -> None:
        env = envpool.make(**env_kwargs)
        super().__init__(env)
        self.num_envs = env_kwargs["num_envs"]
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
        obs, rew, term, trunc, info = self.env.step(actions)

        return obs, rew, term, trunc, info