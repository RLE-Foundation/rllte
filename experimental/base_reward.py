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
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.preprocessing import process_action_space, process_observation_space

class BaseReward(ABC):
    """Base class of reward module.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.

    Returns:
        Instance of the base reward module.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 0.05,
        kappa: float = 0.000025,
    ) -> None:
        # get environment information
        self.obs_shape: Tuple = process_observation_space(observation_space)  # type: ignore
        assert isinstance(self.obs_shape, tuple), "RLLTE currently doesn't support `Dict` observation space!"
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type = process_action_space(action_space)

        # set device and parameters
        self.device = th.device(device)
        self.beta = beta
        self.kappa = kappa
        self.global_step = 0

    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards.
        """
        return self.beta * np.power(1.0 - self.kappa, self.global_step)
    
    @abstractmethod
    def watch(self, 
              observations: th.Tensor,
              actions: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> None:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): The observations.
            actions (th.Tensor): The actions.
            terminateds (th.Tensor): Termination flag.
            truncateds (th.Tensor): Truncation flag.
            next_observations (th.Tensor): The next observations.

        Returns:
            None.
        """
    
    @abstractmethod
    def compute_rewards(self, samples: Dict) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict): The collected samples.

        Returns:
            The intrinsic rewards.
        """

    @abstractmethod
    def update(self, samples: Dict) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict): The collected samples.

        Returns:
            None.
        """
