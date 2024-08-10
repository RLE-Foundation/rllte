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


from typing import Callable, Dict

import gym as gym_old
import gymnasium as gym
import numpy as np
import minihack
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics, EnvCompatibility
from rllte.env.minihack.wrappers import Gym2Gymnasium

from rllte.env.utils import FrameStack, Gymnasium2Torch

def make_minihack_env(
    env_id: str = "MiniHack-MazeWalk-9x9-v0",
    num_envs: int = 8,
    seed: int = 0,
    device: str = "cpu",
    asynchronous: bool = False,
) -> Gymnasium2Torch:
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
        asynchronous (bool): `True` for creating asynchronous environments,
            and `False` for creating synchronous environments.

    Returns:
        The vectorized environments.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym_old.make(env_id)
            env = EnvCompatibility(env)
            env = Gym2Gymnasium(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return _thunk
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]

    if asynchronous:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)

    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device=device)
