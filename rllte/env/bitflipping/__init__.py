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

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from rllte.env.utils import Gymnasium2Torch


class BitFlippingEnv(Env):
    """Simple bit flipping env, useful to test HER.
        The goal is to flip all the bits to get a vector of ones.
        In the continuous variant, if the ith action component has a value > 0,
        then the ith bit will be flipped.

        Borrowed from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/bit_flipping_env.py

    Args:
        n_bits (int): Number of bits to flip
        continuous (bool): Whether to use the continuous actions version or not,
            by default, it uses the discrete one.
        max_steps (int):  Max number of steps, by default, equal to n_bits.
        discrete_obs_space (bool): Whether to use the discrete observation
            version or not, by default, it uses the ``MultiBinary`` one.
        image_obs_space (bool): Use image as input instead of the ``MultiBinary`` one.
        channel_first (bool): Whether to use channel-first or last image.

    Returns:
        Bit flipping environment.
    """

    spec = EnvSpec("BitFlippingEnv-v0", "no-entry-point")
    state: np.ndarray

    def __init__(
        self,
        n_bits: int = 10,
        continuous: bool = False,
        max_steps: Optional[int] = None,
        discrete_obs_space: bool = False,
        image_obs_space: bool = False,
        channel_first: bool = True,
        render_mode: str = "human",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        # shape of the observation when using image space
        self.image_shape = (1, 36, 36) if channel_first else (36, 36, 1)
        # the achieved goal is determined by the current state
        # here, it is a special where they are equal
        if discrete_obs_space:
            # agent act on the binary in the discrete case
            # representation of the observation
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Discrete(2**n_bits),
                    "achieved_goal": spaces.Discrete(2**n_bits),
                    "desired_goal": spaces.Discrete(2**n_bits),
                }
            )
        elif image_obs_space:
            # when using image as input,
            # one image contains the bits 0 -> 0, 1 -> 255
            # and the rest is filled with zeros
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                    "achieved_goal": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                    "desired_goal": spaces.Box(
                        low=0,
                        high=255,
                        shape=self.image_shape,
                        dtype=np.uint8,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.MultiBinary(n_bits),
                    "achieved_goal": spaces.MultiBinary(n_bits),
                    "desired_goal": spaces.MultiBinary(n_bits),
                }
            )

        self.obs_space = spaces.MultiBinary(n_bits)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=(n_bits,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_bits)
        self.continuous = continuous
        self.discrete_obs_space = discrete_obs_space
        self.image_obs_space = image_obs_space
        self.desired_goal = np.ones((n_bits,), dtype=self.observation_space["desired_goal"].dtype)
        if max_steps is None:
            max_steps = n_bits
        self.max_steps = max_steps
        self.current_step = 0

    def seed(self, seed: int) -> None:
        self.obs_space.seed(seed)

    def convert_if_needed(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """Convert to discrete space if needed.

        Args:
            state (np.ndarray): The state to be converted.

        Returns:
            Converted state.
        """
        if self.discrete_obs_space:
            # The internal state is the binary representation of the
            # observed one
            return int(sum(state[i] * 2**i for i in range(len(state))))

        if self.image_obs_space:
            size = np.prod(self.image_shape)
            image = np.concatenate((state * 255, np.zeros(size - len(state), dtype=np.uint8)))
            return image.reshape(self.image_shape).astype(np.uint8)
        return state

    def convert_to_bit_vector(self, state: Union[int, np.ndarray], batch_size: int) -> np.ndarray:
        """Convert to bit vector if needed.

        Args:
            state (Union[int, np.ndarray]): The state to be converted, which can be either an integer or a numpy array.
            batch_size (int): The batch size.

        Returns:
            The state converted into a bit vector.
        """
        # Convert back to bit vector
        if isinstance(state, int):
            bit_vector = np.array(state).reshape(batch_size, -1)
            # Convert to binary representation
            bit_vector = ((bit_vector[:, :] & (1 << np.arange(len(self.state)))) > 0).astype(int)
        elif self.image_obs_space:
            bit_vector = state.reshape(batch_size, -1)[:, : len(self.state)] / 255
        else:
            bit_vector = np.array(state).reshape(batch_size, -1)
        return bit_vector

    def _get_obs(self) -> Dict[str, Union[int, np.ndarray]]:
        """Helper to create the observation."""
        return OrderedDict(
            [
                ("observation", self.convert_if_needed(self.state.copy())),
                ("achieved_goal", self.convert_if_needed(self.state.copy())),
                ("desired_goal", self.convert_if_needed(self.desired_goal.copy())),
            ]
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Union[int, np.ndarray]], Dict]:
        if seed is not None:
            self.obs_space.seed(seed)
        self.current_step = 0
        self.state = self.obs_space.sample()
        return self._get_obs(), {}

    def step(self, action: Union[np.ndarray, int]) -> Tuple[Dict[str, Union[int, np.ndarray]], float, bool, bool, Dict]:
        """Step into the env.

        Args:
            action (Union[np.ndarray, int]): Action to take.

        Returns:
            Time step data.
        """
        if self.continuous:
            self.state[action > 0] = 1 - self.state[action > 0]
        else:
            self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None).item())
        terminated = reward == 0
        self.current_step += 1
        # Episode terminate when we reached the goal or the max number of steps
        info = {"is_success": terminated}
        truncated = self.current_step >= self.max_steps
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self, achieved_goal: Union[int, np.ndarray], desired_goal: Union[int, np.ndarray], _info: Optional[Dict[str, Any]]
    ) -> np.float32:
        # As we are using a vectorized version, we need to keep track of the `batch_size`
        if isinstance(achieved_goal, int):
            batch_size = 1
        elif self.image_obs_space:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 3 else 1
        else:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1

        desired_goal = self.convert_to_bit_vector(desired_goal, batch_size)
        achieved_goal = self.convert_to_bit_vector(achieved_goal, batch_size)

        # Deceptive reward: it is positive only when the goal is achieved
        # Here we are using a vectorized version
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0).astype(np.float32)


def make_bitflipping_env(
    env_id: str = "BitFlippingEnv-v0",
    num_envs: int = 1,
    device: str = "cpu",
    seed: int = 0,
    parallel: bool = True,
    n_bits: int = 15,
    continuous: bool = False,
    max_steps: Optional[int] = 15,
    discrete_obs_space: bool = False,
    image_obs_space: bool = False,
    channel_first: bool = True,
) -> gym.Env:
    """Create bit flipping environment.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device to convert the data.
        seed (int): Random seed.
        parallel (bool): `True` for creating asynchronous environments, and `False`
            for creating synchronous environments.
        n_bits (int): Number of bits to flip
        continuous (bool): Whether to use the continuous actions version or not,
            by default, it uses the discrete one.
        max_steps (int):  Max number of steps, by default, equal to n_bits.
        discrete_obs_space (bool): Whether to use the discrete observation
            version or not, by default, it uses the ``MultiBinary`` one.
        image_obs_space (bool): Use image as input instead of the ``MultiBinary`` one.
        channel_first (bool): Whether to use channel-first or last image.

    Returns:
        The vectorized environments.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = BitFlippingEnv(
                n_bits=n_bits,
                continuous=continuous,
                max_steps=max_steps,
                discrete_obs_space=discrete_obs_space,
                image_obs_space=image_obs_space,
                channel_first=channel_first,
            )
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
