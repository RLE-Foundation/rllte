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



from typing import Dict, Tuple

import gymnasium as gym
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from rllte.common.prototype import BaseReward
from .model import ObservationEncoder, InverseDynamicsModel


class E3B(BaseReward):
    """Exploration via Elliptical Episodic Bonuses (E3B).
        See paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/f4f79698d48bdc1a6dec20583724182b-Paper-Conference.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        use_rms (bool): Use running mean and std for normalization.
        latent_dim (int): The dimension of encoding vectors.
        n_envs (int): The number of parallel environments.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.

    Returns:
        Instance of E3B.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        latent_dim: int = 128,
        lr: float = 0.001,
        use_rms: bool = True,
        obs_rms: bool = False,
        n_envs: int = 1,
        batch_size: int = 64,
        ridge: float = 0.1,
    ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, use_rms, obs_rms)
        
        self.encoder = ObservationEncoder(obs_shape=self.obs_shape,
                                                   latent_dim=latent_dim).to(self.device)
        
        self.im = InverseDynamicsModel(latent_dim=latent_dim,
                                                    action_dim=self.policy_action_dim).to(self.device)

        if self.action_type == "Discrete":
            self.im_loss = nn.CrossEntropyLoss()
        else:
            self.im_loss = nn.MSELoss()

        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        self.im_opt = th.optim.Adam(self.im.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.ridge = ridge
        
        self.cov_inverse = (th.eye(latent_dim) * (1.0 / ridge)).to(self.device)
        self.outer_product_buffer = th.empty(latent_dim, latent_dim).to(self.device)
        self.cov_inverse = self.cov_inverse.repeat(n_envs, 1, 1)
        self.outer_product_buffer = self.outer_product_buffer.repeat(n_envs, 1, 1)

    def watch(self, 
              observations: th.Tensor,
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> None:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): The observations data with shape (n_steps, n_envs, *obs_shape).
            actions (th.Tensor): The actions data with shape (n_steps, n_envs, *action_shape).
            rewards (th.Tensor): The rewards data with shape (n_steps, n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_steps, n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_steps, n_envs).
            next_observations (th.Tensor): The next observations data with shape (n_steps, n_envs, *obs_shape).

        Returns:
            None.
        """

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
        super().compute(samples)
        
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        
        obs_tensor = samples.get("observations").to(self.device)
        terminateds_tensor = samples.get("terminateds").to(self.device)
        truncateds_tensor = samples.get("truncateds").to(self.device)
        
        obs_tensor = self.normalize(obs_tensor)
        
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            for j in range(n_steps):
                h = self.encoder(obs_tensor[j])
                for env_idx in range(n_envs):
                    u = th.mv(self.cov_inverse[env_idx], h[env_idx])
                    b = th.dot(h[env_idx], u).item()
                    intrinsic_rewards[j, env_idx] = b

                    th.outer(u, u, out=self.outer_product_buffer[env_idx])
                    th.add(self.cov_inverse[env_idx], self.outer_product_buffer[env_idx], alpha=-(1./(1. + b)), out=self.cov_inverse[env_idx])
                    
                    if terminateds_tensor[j, env_idx] or truncateds_tensor[j, env_idx]:
                        self.cov_inverse[env_idx] = th.eye(self.latent_dim) * (1.0 / self.ridge)


        self.update(samples)
        return self.scale(intrinsic_rewards)

    def update(self, samples: Dict) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict): The collected samples. A python dict like
                {observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                next_observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        
        obs_tensor = samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        actions_tensor = samples.get("actions").to(self.device).view(-1, *self.action_shape)
        next_obs_tensor = samples.get("next_observations").to(self.device).view(-1, *self.obs_shape)

        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)

        if self.action_type == "Discrete":
            actions_tensor = samples["actions"].view(n_steps * n_envs)
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float()
        else:
            actions_tensor = samples["actions"].view(n_steps * n_envs, -1)

        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size)

        for _idx, batch_data in enumerate(loader):
            obs, actions, next_obs = batch_data
            obs, actions, next_obs = obs.to(self.device), actions.to(self.device), next_obs.to(self.device)

            self.encoder_opt.zero_grad()
            self.im_opt.zero_grad()

            encoded_obs = self.encoder(obs)
            encoded_next_obs = self.encoder(next_obs)

            pred_actions = self.im(encoded_obs, encoded_next_obs)
            im_loss = self.im_loss(pred_actions, actions)
            im_loss.backward()

            self.encoder_opt.step()
            self.im_opt.step()