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


from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from rllte.common.prototype import BaseIntrinsicRewardModule


class Encoder(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None:
        super().__init__()

        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape))
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.linear = nn.Linear(n_flatten, latent_dim)
        else:
            self.trunk = nn.Sequential(nn.Linear(obs_shape[0], 256), nn.ReLU())
            self.linear = nn.Linear(256, latent_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.linear(self.trunk(obs))


class REVD(BaseIntrinsicRewardModule):
    """Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning (REVD).
        See paper: https://openreview.net/pdf?id=V2pw1VYMrDo

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        latent_dim (int): The dimension of encoding vectors.
        alpha (alpha): The order of Rényi divergence.
        k (int): Use the k-th neighbors.
        average_divergence (bool): Use the average of divergence estimation.

    Returns:
        Instance of REVD.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 0.05,
        kappa: float = 0.000025,
        latent_dim: int = 128,
        alpha: float = 0.5,
        k: int = 5,
        average_divergence: bool = False,
    ) -> None:
        super().__init__(observation_space, action_space, device, beta, kappa)
        self.random_encoder = Encoder(
            obs_shape=self._obs_shape,
            action_dim=self._action_dim,
            latent_dim=latent_dim,
        ).to(self._device)

        # freeze the network parameters
        for p in self.random_encoder.parameters():
            p.requires_grad = False

        self.alpha = alpha
        self.k = k
        self.average_divergence = average_divergence

        self.first_update = True
        self.last_encoded_obs = list()

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
        # compute the weighting coefficient of timestep t
        beta_t = self._beta * np.power(1.0 - self._kappa, step)
        num_steps = samples["obs"].size()[0]
        num_envs = samples["obs"].size()[1]
        obs_tensor = samples["obs"].to(self._device)

        intrinsic_rewards = th.zeros(size=(num_steps, num_envs)).to(self._device)

        if self.first_update:
            with th.no_grad():
                for i in range(num_envs):
                    src_feats = self.random_encoder(obs_tensor[:, i])
                    self.last_encoded_obs.append(src_feats)
            self.first_update = False

            return intrinsic_rewards

        with th.no_grad():
            for i in range(num_envs):
                src_feats = self.random_encoder(obs_tensor[:, i])
                dist_intra = th.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                dist_outer = th.linalg.vector_norm(src_feats.unsqueeze(1) - self.last_encoded_obs[i], ord=2, dim=2)

                if self.average_divergence:
                    L = th.kthvalue(dist_intra, 2, dim=1).values.sum() / num_steps
                    for sub_k in range(self.k):
                        D_step_intra = th.kthvalue(dist_intra, sub_k + 1, dim=1).values
                        D_step_outer = th.kthvalue(dist_outer, sub_k + 1, dim=1).values
                        intrinsic_rewards[:, i] += L * th.pow(D_step_outer / (D_step_intra + 1e-11), 1.0 - self.alpha)

                    intrinsic_rewards /= self.k
                else:
                    D_step_intra = th.kthvalue(dist_intra, self.k + 1, dim=1).values
                    D_step_outer = th.kthvalue(dist_outer, self.k + 1, dim=1).values
                    L = th.kthvalue(dist_intra, 2, dim=1).values.sum() / num_steps
                    intrinsic_rewards[:, i] = L * th.pow(D_step_outer / (D_step_intra + 1e-11), 1.0 - self.alpha)

                self.last_encoded_obs[i] = src_feats

        return beta_t * intrinsic_rewards

    def update(self, samples: Dict) -> None:
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
