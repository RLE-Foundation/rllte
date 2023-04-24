from typing import Union, Dict, Any
from abc import ABC, abstractmethod
import gymnasium as gym
from omegaconf import DictConfig
import torch as th

class BaseStorage(ABC):
    """Base class of storage module.

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like 
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Instance of the base storage.
    """
    def __init__(self,
                 observation_space: Union[gym.Space, DictConfig],
                 action_space: Union[gym.Space, DictConfig],
                 device: th.device = 'cpu',
                 ) -> None:
        if isinstance(observation_space, gym.Space) and isinstance(action_space, gym.Space):
            self._obs_shape = observation_space.shape
            if action_space.__class__.__name__ == "Discrete":
                self._action_shape = (int(action_space.n), )
                self._action_type = "Discrete"

            elif action_space.__class__.__name__ == "Box":
                self._action_shape = action_space.shape
                self._action_type = "Box"
            else:
                raise NotImplementedError("Unsupported action type!")
        elif isinstance(observation_space, DictConfig) and isinstance(action_space, DictConfig):
            # by DictConfig
            self._obs_shape = observation_space.shape
            self._action_shape = action_space.shape
            self._action_type = action_space.type
        else:
            raise NotImplementedError("Unsupported observation and action spaces!")

        self._device = th.device(device)
    
    @abstractmethod
    def add(self, *args) -> None:
        """Add sampled transitions into storage.
        """
    
    @abstractmethod
    def sample(self, *args) -> Any:
        """Sample from the storage.
        """