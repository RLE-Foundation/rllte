from typing import Union, Dict
from abc import ABC, abstractmethod
import gymnasium as gym
from omegaconf import DictConfig

import torch as th


class BaseLearner(ABC):
    """Base class of learner.

    Args:
        obs_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'obs_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like 
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

    Returns:
        Base learner instance.
    """

    def __init__(
        self,
        obs_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: th.device,
        feature_dim: int,
        lr: float,
        eps: float,
    ) -> None:
        if isinstance(obs_space, gym.Space) and isinstance(action_space, gym.Space):
            self.obs_shape = obs_space.shape
            if action_space.__class__.__name__ == "Discrete":
                self.action_shape = (int(action_space.n), )
                self.action_type = "Discrete"

            elif action_space.__class__.__name__ == "Box":
                self.action_shape = action_space.shape
                self.action_type = "Box"
            else:
                raise NotImplementedError("Unsupported action type!")
        elif isinstance(obs_space, DictConfig) and isinstance(action_space, DictConfig):
            # by DictConfig
            self.obs_shape = obs_space.shape
            self.action_shape = action_space.shape
            self.action_type = action_space.type
        else:
            raise NotImplementedError("Unsupported observation and action spaces!")
        
        self.device = th.device(device)
        self.feature_dim = feature_dim
        self.lr = lr
        self.eps = eps

        # placeholder for distribution, augmentation, and intrinsic reward function.
        self.encoder = None
        self.encoder_opt = None
        self.dist = None
        self.aug = None
        self.irs = None

    @abstractmethod
    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training

    @abstractmethod
    def update(self, *kwargs) -> Dict[str, float]:
        """Update learner.

        Args:
            Any possible arguments.


        Returns:
            Training metrics such as loss functions.
        """
