from abc import ABC, abstractmethod
from typing import Dict, Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig


class BaseIntrinsicRewardModule(ABC):
    """Base class of intrinsic reward module.

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.

    Returns:
        Instance of the base intrinsic reward module.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str = "cpu",
        beta: float = 0.05,
        kappa: float = 0.000025,
    ) -> None:
        if isinstance(observation_space, gym.Space) and isinstance(action_space, gym.Space):
            self._obs_shape = observation_space.shape
            if action_space.__class__.__name__ == "Discrete":
                self._action_shape = (int(action_space.n),)
                self._action_type = "Discrete"
            elif action_space.__class__.__name__ == "Box":
                self._action_shape = action_space.shape
                self._action_type = "Box"
            elif action_space.__class__.__name__ == "MultiBinary":
                self._action_shape = action_space.shape
                self._action_type = "MultiBinary"
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
        self._beta = beta
        self._kappa = kappa

    @abstractmethod
    def compute_irs(self, samples: Dict, step: int = 0) -> th.Tensor:
        """Compute the intrinsic rewards for current samples.

        Args:
            samples (Dict): The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
            step (int): The global training step.

        Returns:
            The intrinsic rewards.
        """

    @abstractmethod
    def update(
        self,
        samples: Dict,
    ) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            samples: The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.

        Returns:
            None
        """
