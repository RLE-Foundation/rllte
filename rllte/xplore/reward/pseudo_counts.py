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

from rllte.common.prototype import BaseReward
from .model import InverseDynamicsEncoder



class PseudoCounts(BaseReward):
    """Pseudo-counts based on "Never Give Up: Learning Directed Exploration Strategies (NGU)".
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

    Returns:
        Instance of PseudoCounts.
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
        latent_dim: int = 32,
        lr: float = 0.001,
        batch_size: int = 64,
        episodic_memory_size: int = 1000,
        k: int = 10,
        kernel_cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        c: float = 0.001,
        sm: float = 8.0,
        ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, use_rms)
        # set parameters
        self.lr = lr
        self.batch_size = batch_size
        self.k = k
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.c = c
        self.sm = sm

        # build the episodic memory
        self.storage_size = episodic_memory_size
        self.storage = th.zeros(size=(episodic_memory_size, n_envs, latent_dim))
        self.storage_idx = 0
        self.storage_full = False

        # build the encoder and set the loss function
        self.encoder = InverseDynamicsEncoder(
            obs_shape=self.obs_shape,
            action_dim=self.policy_action_dim,
            latent_dim=latent_dim).to(self.device)
        if self.action_type == "Discrete":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()
    
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
        with th.no_grad():
            self.storage[self.storage_idx] = self.encoder.encode(observations)
            self.storage_idx = (self.storage_idx + 1) % self.storage_size

        # update the storage status
        self.storage_full = self.storage_full or self.storage_idx == 0

    def pseudo_counts(self, embeddings: th.Tensor, memory: th.Tensor) -> th.Tensor:
        """Pseudo counts.

        Args:
            embeddings (th.Tensor): Encoded observations.
            memory (th.Tensor): Episodic memory.

        Returns:
            Conut values.
        """
        num_steps = embeddings.size()[0]
        counts = th.zeros(size=(num_steps,))
        for step in range(num_steps):
            dist = th.norm(embeddings[step] - memory, p=2, dim=1).sort().values[: self.k]
            # moving average
            dist = dist / (dist.mean() + 1e-11)
            dist = th.maximum(dist - self.kernel_cluster_distance, th.zeros_like(dist))
            kernel = self.kernel_epsilon / (dist + self.kernel_epsilon)
            s = th.sqrt(kernel.sum()) + self.c

            if th.isnan(s) or s > self.sm:
                counts[step] = 0.0
            else:
                counts[step] = 1.0 / s

        return counts

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
        # get the number of steps and environments
        assert "observations" in samples.keys(), "The key `observations` must be contained in samples!"
        assert "actions" in samples.keys(), "The key `actions` must be contained in samples!"
        assert "next_observations" in samples.keys(), "The key `next_observations` must be contained in samples!"
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        obs_tensor = samples.get("observations").to(self.device)

        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            for i in range(n_envs):
                embeddings = self.encoder.encode(obs_tensor[:, i])
                episodic_memory = self.storage[:self.storage_idx, i] if not self.storage_full else self.storage[:, i]
                n_eps = self.pseudo_counts(embeddings=embeddings, memory=episodic_memory)
                intrinsic_rewards[:, i] = n_eps
            
        # update the embedding network
        self.update(samples)
        
        # scale the intrinsic rewards
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
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        obs_tensor = samples.get("observations").to(self.device).view((n_steps * n_envs, *self.obs_shape))
        next_obs_tensor = samples.get("next_observations").to(self.device).view((n_steps * n_envs, *self.obs_shape))

        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs).to(self.device)
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float()
        else:
            actions_tensor = samples.get("actions").view((n_steps * n_envs, self.action_dim)).to(self.device)

        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for _idx, batch in enumerate(loader):
            obs, actions, next_obs = batch
            pred_actions = self.encoder(obs, next_obs)
            self.opt.zero_grad()
            loss = self.loss(pred_actions, actions)
            loss.backward()
            self.opt.step()