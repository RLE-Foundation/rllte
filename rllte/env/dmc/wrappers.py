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
from typing import Any, Dict, Tuple

import dm_env
import gymnasium as gym
import numpy as np
from dm_env import specs
from gymnasium import spaces


class ActionRepeatWrapper(dm_env.Environment):
    """Repeats the action for a given number of steps.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/dmc.py

    Args:
        env (dm_env.Environment): Environment to wrap.
        num_repeats (int): Number of times to repeat the action.

    Returns:
        Wrapped environment.
    """

    def __init__(self, env: dm_env.Environment, num_repeats: int) -> None:
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Repeat the action for a given number of steps and return the accumulated reward.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            dm_env.TimeStep: Time step.
        """
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self) -> specs.BoundedArray:
        """Observation spec."""
        return self._env.observation_spec()

    def action_spec(self) -> specs.BoundedArray:
        """Action spec."""
        return self._env.action_spec()

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment."""
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    """Stacks consecutive frames together to feed them to the agent.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/dmc.py

    Args:
        env (dm_env.Environment): Environment to wrap.
        num_frames (int): Number of frames to stack.
        pixels_key (str): Key of the pixels in the observation dictionary.

    Returns:
        Wrapped environment.
    """

    def __init__(self, env: dm_env.Environment, num_frames: int, pixels_key: str = "pixels") -> None:
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate([[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step: dm_env.TimeStep) -> dm_env.TimeStep:
        """Concatenate the frames and return the new observation.

        Args:
            time_step (dm_env.TimeStep): Time step.

        Returns:
            dm_env.TimeStep: Time step with the new observation.
        """
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step: dm_env.TimeStep) -> np.ndarray:
        """Extract pixels from the observation dictionary.

        Args:
            time_step (dm_env.TimeStep): Time step.

        Returns:
            np.ndarray: Pixels.
        """
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment and stack the first frame."""
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Take a step in the environment and stack the new frame.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            dm_env.TimeStep: Time step.
        """
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self) -> specs.BoundedArray:
        """Observation spec."""
        return self._obs_spec

    def action_spec(self) -> specs.BoundedArray:
        """Action spec."""
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    """A wrapper that converts actions to a given dtype.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/dmc.py

    Args:
        env (dm_env.Environment): An environment to be wrapped.
        dtype (np.dtype): A dtype to convert actions to.

    Returns:
        Wrapped environment.
    """

    def __init__(self, env: dm_env.Environment, dtype: np.dtype) -> None:
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape, dtype, wrapped_action_spec.minimum, wrapped_action_spec.maximum, "action"
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Take a step in the environment.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            dm_env.TimeStep: Time step.
        """
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self) -> specs.BoundedArray:
        """Observation spec."""
        return self._env.observation_spec()

    def action_spec(self) -> specs.BoundedArray:
        """Action spec."""
        return self._action_spec

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment."""
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlatObsWrapper(dm_env.Environment):
    """Flattens the observation dictionary into a single array.

    Args:
        env (dm_env.Environment): Environment to wrap.

    Returns:
        Wrapped environment.
    """

    def __init__(self, env: dm_env.Environment) -> None:
        self._env = env

        state_dim = 0
        for item in env.observation_spec().values():
            state_dim += item.shape[0]

        self._obs_spec = specs.BoundedArray(
            shape=(state_dim,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="observation"
        )

    def _flatten_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten the observation dictionary into a single array.

        Args:
            obs (Dict[str, np.ndarray]): Observation dictionary.

        Returns:
            np.ndarray: Flattened observation.
        """
        obs_pieces = []
        for v in obs.values():
            flat = np.array([v]) if np.isscalar(v) else v.ravel()
            obs_pieces.append(flat)
        return np.concatenate(obs_pieces, axis=0, dtype=np.float32)

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment and flatten the observation."""
        time_step = self._env.reset()
        return time_step._replace(observation=self._flatten_obs(time_step.observation))

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Take a step in the environment and flatten the observation."""
        time_step = self._env.step(action)
        return time_step._replace(observation=self._flatten_obs(time_step.observation))

    def observation_spec(self) -> specs.BoundedArray:
        """Observation spec."""
        return self._obs_spec

    def action_spec(self) -> specs.BoundedArray:
        """Action spec."""
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class DMC2Gymnasium(gym.Env):
    """A wrapper for gymnasium interface of DeepMind Control Suite.

    Args:
        env (dm_env.Environment): An environment to be wrapped.

    Returns:
        Wrapped environment.
    """

    def __init__(self, env: dm_env.Environment) -> None:
        super().__init__()
        self.env = env

        self.observation_space = self._spec_to_box(self.env.observation_spec())
        self.action_space = self._spec_to_box(self.env.action_spec())

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Action to take.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation, reward, terminated, truncated, info.
        """
        time_step = self.env.step(action)
        obs = time_step.observation
        reward = time_step.reward
        terminated = False  # never stop
        truncated = time_step.last()
        info = {"discount": time_step.discount}

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment."""
        time_step = self.env.reset()
        obs = time_step.observation
        info = {"discount": time_step.discount}
        return obs, info

    def _spec_to_box(self, spec: specs.Array) -> spaces.Box:
        """Converts a dm_env spec to a gym Box space.

        Args:
            spec (specs.Array): A dm_env spec.

        Returns:
            A gym Box space.
        """
        shape = spec.shape
        low_value = spec.minimum * np.ones(shape, dtype=spec.dtype)
        high_value = spec.maximum * np.ones(shape, dtype=spec.dtype)

        return spaces.Box(low=low_value, high=high_value, shape=spec.shape, dtype=spec.dtype)
