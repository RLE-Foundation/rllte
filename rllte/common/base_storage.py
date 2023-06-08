from abc import ABC, abstractmethod
from typing import Any, Union

import gymnasium as gym
import torch as th

class BaseStorage(ABC):
    """Base class of storage module.

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment. 
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
        self.obs_shape = observation_space.shape
        if action_space.__class__.__name__ == "Discrete":
            self.action_shape = action_space.shape
            self.action_dim = int(action_space.n)
            self.action_type = "Discrete"
            self.action_range = [0, int(action_space.n) - 1]

        elif action_space.__class__.__name__ == "Box":
            self.action_shape = action_space.shape
            self.action_dim = action_space.shape[0]
            self.action_type = "Box"
            self.action_range = [
                float(action_space.low[0]),
                float(action_space.high[0]),
            ]
            
        elif action_space.__class__.__name__ == "MultiBinary":
            self.action_shape = action_space.shape
            self.action_dim = action_space.shape[0]
            self.action_type = "MultiBinary"
            self.action_range = [0, 1]
        else:
            raise NotImplementedError("Unsupported action type!")

        self.device = th.device(device)

    @abstractmethod
    def add(self, *args) -> None:
        """Add sampled transitions into storage."""

    @abstractmethod
    def sample(self, *args) -> Any:
        """Sample from the storage."""

    @abstractmethod
    def update(self, *args) -> None:
        """Update the storage if necessary."""
