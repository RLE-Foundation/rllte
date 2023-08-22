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


from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.preprocessing import process_observation_space, process_action_space

class BaseStorage(ABC):
    """Base class of the storage module.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        storage_size (int): The size of the storage.
        batch_size (int): Batch size of samples.
        num_envs (int): The number of parallel environments.

    Returns:
        Instance of the base storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str,
        storage_size: int,
        batch_size: int,
        num_envs: int
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.storage_size = storage_size
        self.batch_size = batch_size
        self.num_envs = num_envs
        # get environment information
        self.obs_shape = process_observation_space(observation_space)
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type = process_action_space(action_space)
        # set device
        self.device = th.device(device)
        # counter
        self.step = 0
        self.full = False

    def to_torch(self, x: np.ndarray) -> th.Tensor:
        """Convert numpy array to torch tensor.

        Args:
            x (np.ndarray): Numpy array.

        Returns:
            Torch tensor.
        """
        return th.as_tensor(x, device=self.device).float()
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the storage."""
        self.step = 0
        self.full = False

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """Add samples to the storage."""

    @abstractmethod
    def sample(self, *args, **kwargs) -> Any:
        """Sample from the storage."""

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the storage if necessary."""
