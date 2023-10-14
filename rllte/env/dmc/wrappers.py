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

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from dm_control import manipulation, suite
from dm_env import TimeStep, specs
from gymnasium import spaces


class DMC2Gymnasium(gym.Env):
    """A gymnasium wrapper for dm_control environments.

    Args:
        env_id (str): Name of environment.
        seed (int): Random seed.
        visualize_reward (bool): Opposite to `from_pixels`.
        from_pixels (bool): Provide image-based observations or not.
        height (int): Image observation height.
        width (int): Image observation width.
        frame_stack (int): Number of stacked frames.
        action_repeat (int): Number of action repeats.
    """

    def __init__(
        self,
        env_id: str,
        seed=1,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        frame_stack=3,
        action_repeat=1,
    ) -> None:
        assert visualize_reward != from_pixels, "`visualize_reward` and `from_pixels` cannot be both True or False!"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._action_repeat = action_repeat

        domain, task = env_id.split("_", 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup="ball_in_cup").get(domain, domain)
        # create task
        if (domain, task) in suite.ALL_TASKS:
            self._env = suite.load(domain, task, task_kwargs={"random": seed}, visualize_reward=False)
        else:
            name = f"{domain}_{task}_vision"
            self._env = manipulation.load(name, seed=seed)
        # zoom in for quadruped task
        self._camera_id = dict(quadruped=2).get(domain, 0)

        # create true and normalized action spaces
        self._true_action_space = self._spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32)

        # create observation and state space
        if from_pixels:
            self._observation_space = spaces.Box(low=0, high=255, shape=[3, height, width], dtype=np.uint8)
        else:
            self._observation_space = self._spec_to_box(self._env.observation_spec().values(), np.float32)

        self._state_space = self._spec_to_box(self._env.observation_spec().values(), np.float32)

        # set seed
        self.seed(seed=seed)

    def _get_obs(self, time_step: TimeStep) -> np.ndarray:
        """Get observations from dm_control environment.

        Args:
            time_step (TimeStep): A dm_control time step.

        Returns:
            Observations.
        """
        if self._from_pixels:
            obs = self.render(height=self._height, width=self._width, camera_id=self._camera_id)
            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = self._flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """Converts actions from normalized to true action space."""
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Steps the environment.

        Args:
            action (np.ndarray): A normalized action.

        Returns:
            Next observation, reward, termination, truncation, and extra information.
        """
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            truncation = time_step.last()  # never stop
            if truncation:
                break
        obs = self._get_obs(time_step)
        self.current_state = self._flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, False, truncation, extra

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        time_step = self._env.reset()
        self.current_state = self._flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs, {}

    def render(
        self,
        mode: str = "rgb_array",
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_id: Optional[int] = None,
    ) -> np.ndarray:
        """Render the environment.

        Args:
            mode (str): Rendering mode.
            height (Optional[int]): Image height.
            width (Optional[int]): Image width.
            camera_id (Optional[int]): Camera id.

        Returns:
            An image of the environment.
        """
        assert mode == "rgb_array", "only support rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed: int) -> None:
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def _flatten_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Extracts the values from a dm_control observation dict and concatenates them.

        Args:
            obs (Dict[str, np.ndarray]): A dm_control observation dict.

        Returns:
            A flattened numpy array containing the values of the observation dict.
        """
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0)

    def _spec_to_box(self, spec: specs.Array, dtype: np.dtype) -> spaces.Box:
        """Transforms a dm_control Array spec into a gymnasium Box space.

        Args:
            spec (specs.Array): The dm_control Array spec.
            dtype (np.dtype): The dtype of the resulting `Box` space.

        Returns:
            A gymnasium Box space with the same shape as the spec and the given dtype.
        """

        def _extract_min_max(s):
            assert s.dtype == np.float64 or s.dtype == np.float32
            dim = int(np.prod(s.shape))
            if type(s) == specs.Array:
                bound = np.inf * np.ones(dim, dtype=np.float32)
                return -bound, bound
            elif type(s) == specs.BoundedArray:
                zeros = np.zeros(dim, dtype=np.float32)
                return s.minimum + zeros, s.maximum + zeros

        mins, maxs = [], []
        for s in spec:
            mn, mx = _extract_min_max(s) # type: ignore
            mins.append(mn)
            maxs.append(mx)
        low = np.concatenate(mins, axis=0).astype(dtype)
        high = np.concatenate(maxs, axis=0).astype(dtype)
        assert low.shape == high.shape
        return spaces.Box(low, high, dtype=dtype)
