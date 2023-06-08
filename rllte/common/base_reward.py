from abc import ABC, abstractmethod
from typing import Dict

import gymnasium as gym
import torch as th


class BaseIntrinsicRewardModule(ABC):
    """Base class of intrinsic reward module.

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.

    Returns:
        Instance of the base intrinsic reward module.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 0.05,
        kappa: float = 0.000025,
    ) -> None:
        self._obs_shape = observation_space.shape
        if action_space.__class__.__name__ == "Discrete":
            self._action_shape = action_space.shape
            self._action_dim = action_space.n
            self._action_type = "Discrete"
        elif action_space.__class__.__name__ == "Box":
            self._action_shape = action_space.shape
            self._action_dim = action_space.shape[0]
            self._action_type = "Box"
        elif action_space.__class__.__name__ == "MultiBinary":
            self._action_shape = action_space.shape
            self._action_dim = action_space.shape[0]
            self._action_type = "MultiBinary"
        else:
            raise NotImplementedError("Unsupported action type!")

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
