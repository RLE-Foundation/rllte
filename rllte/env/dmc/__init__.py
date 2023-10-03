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


from typing import Callable

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.dmc.wrappers import DMC2Gymnasium
from rllte.env.utils import FrameStack, Gymnasium2Torch


def make_dmc_env(
    env_id: str = "humanoid_run",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 1,
    visualize_reward: bool = True,
    from_pixels: bool = False,
    height: int = 84,
    width: int = 84,
    frame_stack: int = 3,
    action_repeat: int = 1,
    asynchronous: bool = True,
) -> Gymnasium2Torch:
    """Create DeepMind Control Suite environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        visualize_reward (bool): Opposite to `from_pixels`.
        from_pixels (bool): Provide image-based observations or not.
        height (int): Image observation height.
        width (int): Image observation width.
        frame_stack (int): Number of stacked frames.
        action_repeat (int): Number of action repeats.
        asynchronous (bool): `True` for creating asynchronous environments,
            and `False` for creating synchronous environments.

    Returns:
        The vectorized environments.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            # convert to gym interface
            env = DMC2Gymnasium(
                env_id=env_id,
                seed=seed,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                frame_stack=frame_stack,
                action_repeat=action_repeat,
            )
            if from_pixels:
                env = FrameStack(env, frame_stack)
            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    if asynchronous:
        envs = AsyncVectorEnv(envs)
    else:
        envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device)
