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


from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper

from rllte.env.utils import FrameStack, TorchVecEnvWrapper


class Minigrid2Image(gym.ObservationWrapper):
    """Convert MiniGrid observation to image.

    Args:
        env (gym.Env): Environment to wrap.

    Returns:
        Minigrid2Image instance.
    """

    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space["image"]
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[2], shape[0], shape[1]),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation: Dict) -> np.ndarray:
        """Convert MiniGrid observation to image."""
        return np.transpose(observation["image"], axes=[2, 0, 1])


def make_minigrid_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    num_envs: int = 8,
    fully_observable: bool = True,
    seed: int = 0,
    frame_stack: int = 1,
    device: str = "cpu",
    distributed: bool = False,
) -> gym.Env:
    """Build MiniGrid environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        fully_observable (bool): 'True' for using fully observable RGB image as observation.
        seed (int): Random seed.
        frame_stack (int): Number of stacked frames.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        distributed (bool): For `Distributed` algorithms, in which `SyncVectorEnv` is required
            and reward clip will be used before environment vectorization.

    Returns:
        The vectorized environment.
    """

    def make_env(env_id: str, seed: int) -> gym.Env:
        def _thunk():
            env = gym.make(env_id)

            if fully_observable:
                env = FullyObsWrapper(env)
                env = Minigrid2Image(env)
                env = FrameStack(env, k=frame_stack)
            else:
                env = FlatObsWrapper(env)

            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    if distributed:
        envs = SyncVectorEnv(envs)
    else:
        envs = AsyncVectorEnv(envs)
        envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device=device)
