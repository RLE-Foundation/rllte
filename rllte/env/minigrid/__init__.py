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
from minigrid.wrappers import DictObservationSpaceWrapper, FlatObsWrapper, FullyObsWrapper

from rllte.env.utils import FrameStack, Gymnasium2Torch


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


class ImageTranspose(gym.ObservationWrapper):
    """Transpose observation from channels last to channels first.

    Args:
        env (gym.Env): Environment to wrap.

    Returns:
        Minigrid2Image instance.
    """

    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shape = env.observation_space["image"].shape
        dtype = env.observation_space["image"].dtype
        self.observation_space["image"] = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )

    def observation(self, observation: Dict) -> Dict[str, np.ndarray]:
        """Convert MiniGrid observation to image."""
        observation["image"] = np.transpose(observation["image"], axes=[2, 0, 1])
        return observation


def make_minigrid_env(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    num_envs: int = 8,
    fully_observable: bool = True,
    fully_numerical: bool = False,
    seed: int = 0,
    frame_stack: int = 1,
    device: str = "cpu",
    parallel: bool = True,
) -> gym.Env:
    """Create MiniGrid environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        fully_observable (bool): Fully observable gridworld using a compact grid encoding instead of the agent view.
        fully_numerical (bool): Transforms the observation space (that has a textual component) to a fully numerical
            observation space, where the textual instructions are replaced by arrays representing the indices of each
            word in a fixed vocabulary.
        seed (int): Random seed.
        frame_stack (int): Number of stacked frames.
        device (str): Device to convert the data.
        parallel (bool): `True` for creating asynchronous environments, and `False`
            for creating synchronous environments.

    Returns:
        The vectorized environments.
    """

    def make_env(env_id: str, seed: int) -> gym.Env:
        def _thunk():
            env = gym.make(env_id)

            if fully_observable:
                env = FullyObsWrapper(env)
                env = Minigrid2Image(env)
                env = FrameStack(env, k=frame_stack)
            elif fully_numerical:
                env = DictObservationSpaceWrapper(env)
                env = ImageTranspose(env)
            else:
                env = FlatObsWrapper(env)

            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]

    if parallel:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device=device)
