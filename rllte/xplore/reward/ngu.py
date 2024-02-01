# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

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


from typing import Dict
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym
import torch as th

from .fabric import Fabric
from .pseudo_counts import PseudoCounts
from .rnd import RND

class NGU(Fabric):
    """Never Give Up: Learning Directed Exploration Strategies (NGU).
        See paper: https://arxiv.org/pdf/2002.06038

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        n_envs (int): The number of parallel environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        use_rms (bool): Use running mean and std for normalization.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for update.
        episodic_memory_size (int): The capacity of the episodic memory.
        k (int): Number of neighbors.
        kernel_cluster_distance (float): The kernel cluster distance.
        kernel_epsilon (float): The kernel constant.
        c (float): The pseudo-counts constant.
        sm (float): The kernel maximum similarity.
        mrs (float): The maximum reward scaling.

    Returns:
        Instance of NGU.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        use_rms: bool = True,
        obs_rms: bool = False,
        latent_dim: int = 32,
        lr: float = 0.001,
        batch_size: int = 64,
        episodic_memory_size: int = 1000,
        k: int = 10,
        kernel_cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        c: float = 0.001,
        sm: float = 8.0,
        mrs: float = 5.0,
        update_proportion: float = 1.0
        ) -> None:

        rnd = RND(observation_space=observation_space,
                  action_space=action_space,
                  n_envs=n_envs,
                  device=device,
                  beta=beta,
                  kappa=kappa,
                  use_rms=use_rms,
                  obs_rms=obs_rms,
                  latent_dim=latent_dim,
                  lr=lr,
                  batch_size=batch_size,
                  update_proportion=update_proportion
                )
        
        pseudo_counts = PseudoCounts(observation_space=observation_space,
                                     action_space=action_space,
                                     n_envs=n_envs,
                                     device=device,
                                     beta=beta,
                                     kappa=kappa,
                                     use_rms=use_rms,
                                     obs_rms=obs_rms,
                                     latent_dim=latent_dim,
                                     lr=lr,
                                     batch_size=batch_size,
                                     k=k,
                                     kernel_cluster_distance=kernel_cluster_distance,
                                     kernel_epsilon=kernel_epsilon,
                                     c=c,
                                     sm=sm,
                                     update_proportion=update_proportion)

        super().__init__(*[rnd, pseudo_counts])

    def compute(self, samples: Dict) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict): The collected samples. A python dict like
                {observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                next_observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
                The derived intrinsic rewards have the shape of (n_steps, n_envs).

        Returns:
            The intrinsic rewards.
        """
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        lifelong_rewards, episodic_rewards = super().compute(samples)

        # compute the intrinsic rewards
        lifelong_rewards = 1.0 + lifelong_rewards
        lifelong_rewards = th.maximum(lifelong_rewards, th.ones_like(lifelong_rewards))
        lifelong_rewards = th.minimum(lifelong_rewards, th.ones_like(lifelong_rewards) * self.mrs)
        
        return lifelong_rewards * episodic_rewards