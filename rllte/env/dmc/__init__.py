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

import gymnasium as gym
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.dmc.wrappers import ActionDTypeWrapper, ActionRepeatWrapper, DMC2Gymnasium, FlatObsWrapper, FrameStackWrapper
from rllte.env.utils import Gymnasium2Torch


def make_dmc_env(
    env_id: str = "cartpole_balance",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 1,
    visualize_reward: bool = False,
    from_pixels: bool = True,
    height: int = 84,
    width: int = 84,
    frame_stack: int = 3,
    action_repeat: int = 2,
) -> gym.Env:
    """Create DeepMind Control Suite environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        visualize_reward (bool): True when 'from_pixels' is False, False when 'from_pixels' is True.
        from_pixels (bool): Provide image-based observations or not.
        height (int): Image observation height.
        width (int): Image observation width.
        frame_stack (int): Number of stacked frames.
        action_repeat (int): Number of action repeats.

    Returns:
        The vectorized environments.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            domain, task = env_id.split("_", 1)
            # overwrite cup to ball_in_cup
            domain = dict(cup="ball_in_cup").get(domain, domain)
            if from_pixels:
                assert not visualize_reward, "Cannot use visualize reward when learning from pixels!"
            if (domain, task) in suite.ALL_TASKS:
                env = suite.load(domain, task, task_kwargs={"random": seed}, visualize_reward=False)
                pixels_key = "pixels"
            else:
                name = f"{domain}_{task}_vision"
                env = manipulation.load(name, seed=seed)
                pixels_key = "front_close"
            # add wrappers
            env = ActionDTypeWrapper(env, np.float32)
            env = ActionRepeatWrapper(env, action_repeat if from_pixels else 1)
            env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

            if visualize_reward and not from_pixels:
                env = FlatObsWrapper(env)
            else:
                # add renderings for clasical tasks
                if (domain, task) in suite.ALL_TASKS:
                    # zoom in camera for quadruped
                    camera_id = dict(quadruped=2).get(domain, 0)
                    render_kwargs = dict(height=height, width=width, camera_id=camera_id)
                    env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
                # stack several frames
                env = FrameStackWrapper(env, frame_stack, pixels_key)

            # convert to gym interface
            env = DMC2Gymnasium(env)

            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return Gymnasium2Torch(envs, device)
