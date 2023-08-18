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

from rllte.common.preprocessing import process_env_info

class BaseStorage(ABC):
    """Base class of storage module.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Instance of the base storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        # get environment information
        self.obs_shape, self.action_shape, self.action_dim, self.action_type, self.action_range = process_env_info(
            observation_space, action_space
        )
        # set device
        self.device = th.device(device)

    def to_torch(self, x: np.ndarray) -> th.Tensor:
        """Convert numpy array to torch tensor.

        Args:
            x (np.ndarray): Numpy array.

        Returns:
            Torch tensor.
        """
        return th.as_tensor(x, device=self.device).float()
    
    @abstractmethod
    def add(self, *args) -> None:
        """Add samples to the storage."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the storage."""

    @abstractmethod
    def sample(self, *args) -> Any:
        """Sample from the storage."""

    @abstractmethod
    def update(self, *args) -> None:
        """Update the storage if necessary."""
