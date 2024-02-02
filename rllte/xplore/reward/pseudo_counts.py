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


from typing import Dict, List, Optional
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
        kappa (float): The decay rate of the weighting coefficient.
        rwd_rms (bool): Use running mean and std for normalization.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for update.
        k (int): Number of neighbors.
        kernel_cluster_distance (float): The kernel cluster distance.
        kernel_epsilon (float): The kernel constant.
        c (float): The pseudo-counts constant.
        sm (float): The kernel maximum similarity.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.

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
        rwd_rms: bool = True,
        obs_rms: bool = False,
        latent_dim: int = 32,
        lr: float = 0.001,
        batch_size: int = 64,
        k: int = 10,
        kernel_cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        c: float = 0.001,
        sm: float = 8.0,
        update_proportion: float = 1.0,
        ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, rwd_rms, obs_rms)
        # set parameters
        self.lr = lr
        self.batch_size = batch_size
        self.k = k
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.c = c
        self.sm = sm
        self.update_proportion = update_proportion

        # build the episodic memory
        self.episodic_memory = [[] for _ in range(n_envs)]
        self.n_eps = [[] for _ in range(n_envs)]

        # build the encoder
        self.encoder = InverseDynamicsEncoder(
            obs_shape=self.obs_shape,
            action_dim=self.policy_action_dim,
            latent_dim=latent_dim).to(self.device)
        # set the optimizer and loss function
        self.opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        if self.action_type == "Discrete":
            self.loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss = nn.MSELoss(reduction="none")
    
    def watch(self, 
              observations: th.Tensor, 
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples, e.g., intrinsic rewards for the current samples. This 
            is useful when applying the memory-based methods to off-policy algorithms.
        """
        with th.no_grad():
            # data shape of embeddings: (n_envs, latent_dim)
            observations = self.normalize(observations)
            embeddings = self.encoder.encode(observations)
            for i in range(self.n_envs):
                # update the episodic memory
                self.episodic_memory[i].append(embeddings[i])            
                n_eps = self.pseudo_counts(embeddings=embeddings[i].unsqueeze(0), memory=self.episodic_memory[i])
                # store the pseudo-counts
                self.n_eps[i].append(n_eps)
                # clear the episodic memory if the episode is terminated or truncated
                if terminateds[i].item() or truncateds[i].item():
                    self.episodic_memory[i].clear()
                    # print(terminateds, truncateds)
        
        # if self.is_sync:
        #     return {'single_step_rewards': th.as_tensor(n_eps)}
        # else:
        #     return None

    def pseudo_counts(self, embeddings: th.Tensor, memory: List[th.Tensor]) -> th.Tensor:
        """Pseudo counts.

        Args:
            embeddings (th.Tensor): Encoded observations.
            memory (List[th.Tensor]): Episodic memory.

        Returns:
            Conut values.
        """
        memory = th.stack(memory)
        dist = th.norm(embeddings - memory, p=2, dim=1).sort().values[: self.k]
        # moving average
        dist = dist / (dist.mean() + 1e-11)
        dist = th.maximum(dist - self.kernel_cluster_distance, th.zeros_like(dist))
        kernel = self.kernel_epsilon / (dist + self.kernel_epsilon)
        s = th.sqrt(kernel.sum()) + self.c

        if th.isnan(s) or s > self.sm:
            return 0.0
        else:
            return 1.0 / s

    def compute(self, samples: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors, whose keys are
            'observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'. For example, 
            the data shape of 'observations' is (n_steps, n_envs, *obs_shape). 

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)

        # compute the intrinsic rewards
        all_n_eps = [th.as_tensor(n_eps) for n_eps in self.n_eps]
        intrinsic_rewards = th.stack(all_n_eps).T.to(self.device)

        # flush the episodic memory of intrinsic rewards
        self.n_eps = [[] for _ in range(self.n_envs)]
            
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
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        obs_tensor = samples.get("observations").to(self.device).view((n_steps * n_envs, *self.obs_shape))
        next_obs_tensor = samples.get("next_observations").to(self.device).view((n_steps * n_envs, *self.obs_shape))
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs).to(self.device)
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float()
        else:
            actions_tensor = samples.get("actions").view((n_steps * n_envs, self.action_dim)).to(self.device)
        # build the dataset and loader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        # update the encoder
        for _idx, batch in enumerate(loader):
            # get the batch data
            obs, actions, next_obs = batch
            # zero the gradients
            self.opt.zero_grad()
            # get the predicted actions
            pred_actions = self.encoder(obs, next_obs)
            # compute the inverse dynamics loss
            im_loss = self.loss(pred_actions, actions)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(im_loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # get the masked loss
            im_loss = (im_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            im_loss.backward()
            self.opt.step()