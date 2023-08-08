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


from typing import Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from robosuite.wrappers import GymWrapper

from rllte.env.utils import Gymnasium2Torch


class AdapterEnv(gym.Wrapper):
    """PyBullet robotics envs currently doesn't support Gymnasium.

    Args:
        env (gym.Env): Environment to wrap.

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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            Observation, reward, terminated, truncated, and info.
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, done, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        obs = self.env.reset()
        return obs, {}


def make_robosuite_env(
    env_id: str = "Lift_Panda",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 0,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = False,
    use_camera_obs: bool = False,
    parallel: bool = True,
) -> gym.Env:
    """Build Robosuite robotics environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        seed (int): Random seed.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering.
        use_camera_obs (bool): True for using image observations.
        parallel (bool): `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.
            For `Distributed` algorithms, in which `SyncVectorEnv` is required
            and reward clip will be used before environment vectorization.

    Returns:
        The vectorized environment.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env_name, robots = env_id.split("_")
            env = suite.make(
                env_name=env_name,
                robots=robots,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                use_camera_obs=use_camera_obs,
            )
            env = GymWrapper(env)
            env.seed(seed)
            env = AdapterEnv(env)
            env.observation_space.seed(seed)
            env.action_space.seed(seed)
            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]

    if parallel:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device=device)
