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



from typing import List, Dict, Optional

import gymnasium as gym
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from rllte.common.utils import TorchRunningMeanStd
from rllte.common.prototype import BaseReward
from .model import ObservationEncoder, InverseDynamicsModel, ForwardDynamicsModel

from rllte.xploit.encoder import MinigridEncoder

class RIDE(BaseReward):
    """RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments.
        See paper: https://arxiv.org/pdf/2002.12292

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        rwd_norm_type (bool): Use running mean and std for reward normalization.
        obs_rms (bool): Use running mean and std for observation normalization.
        gamma (Optional[float]): Intrinsic reward discount rate, None for no discount.
        latent_dim (int): The dimension of encoding vectors.
        n_envs (int): The number of parallel environments.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.

    Returns:
        Instance of RIDE.
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
        rwd_norm_type: str = "rms",
        obs_rms: bool = False,
        gamma: Optional[float] = None,
        n_envs: int = 1,
        batch_size: int = 256,
        # episodic memory
        k: int = 10,
        kernel_cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        c: float = 0.001,
        sm: float = 8.0,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "default"
    ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, rwd_norm_type, obs_rms, gamma)
        # build the encoder, inverse dynamics model and forward dynamics model
        self.encoder = MinigridEncoder(observation_space=observation_space).to(self.device)
        self.im = InverseDynamicsModel(latent_dim=latent_dim, 
                                       action_dim=self.policy_action_dim, encoder_model=encoder_model, weight_init=weight_init).to(self.device)
        self.fm = ForwardDynamicsModel(latent_dim=latent_dim, 
                                       action_dim=self.policy_action_dim, encoder_model=encoder_model, weight_init=weight_init).to(self.device)
        # set the loss function
        if self.action_type == "Discrete":
            self.im_loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self.im_loss = nn.MSELoss(reduction="none")
        # set the optimizers
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        self.im_opt = th.optim.Adam(self.im.parameters(), lr=lr)
        self.fm_opt = th.optim.Adam(self.fm.parameters(), lr=lr)
        # set the parameters
        self.batch_size = batch_size
        self.update_proportion = update_proportion
        self.k = k
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.c = c
        self.sm = sm
        # build the episodic memory
        self.episodic_memory = [[] for _ in range(n_envs)]
        self.n_eps = [[] for _ in range(n_envs)]
        
        # rms for the intrinsic rewards
        self.dist_rms = TorchRunningMeanStd(shape=(1, ), device=self.device)
        self.squared_distances = []

    def pseudo_counts(self, embeddings: th.Tensor, memory: List[th.Tensor]) -> th.Tensor:
        """Pseudo counts.

        Args:
            embeddings (th.Tensor): Encoded observations.
            memory (List[th.Tensor]): Episodic memory.

        Returns:
            Conut values.
        """
        memory = th.stack(memory)
        dist = (th.norm(embeddings - memory, p=2, dim=1).sort().values[: self.k]) ** 2
        self.squared_distances.append(dist)
        dist = dist / (self.dist_rms.mean + 1e-8)
        dist = th.maximum(dist - self.kernel_cluster_distance, th.zeros_like(dist))
        kernel = self.kernel_epsilon / (dist + self.kernel_epsilon)
        s = th.sqrt(kernel.sum()) + self.c
        
        if th.isnan(s) or s > self.sm:
            return 0.0
        else:
            return 1.0 / s

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
            embeddings = self.encoder(observations)
            for i in range(self.n_envs):
                if len(self.episodic_memory[i]) > 0:
                    # compute pseudo-counts
                    n_eps = self.pseudo_counts(embeddings=embeddings[i].unsqueeze(0), memory=self.episodic_memory[i])
                else:
                    n_eps = 0.0
                # store the pseudo-counts
                self.n_eps[i].append(n_eps)
                # update the episodic memory
                self.episodic_memory[i].append(embeddings[i])
                # clear the episodic memory if the episode is terminated or truncated
                if terminateds[i].item() or truncateds[i].item():
                    self.episodic_memory[i].clear()
        
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
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the observations and next observations
        obs_tensor = samples.get("observations").to(self.device)
        next_obs_tensor = samples.get("next_observations").to(self.device)
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            for i in range(self.n_envs):
                encoded_obs = self.encoder(obs_tensor[:, i])
                encoded_next_obs = self.encoder(next_obs_tensor[:, i])                
                dist = F.mse_loss(encoded_obs, encoded_next_obs, reduction="none").mean(dim=1)

                intrinsic_rewards[:, i] = dist.cpu()

        # get all the n_eps
        all_n_eps = [th.as_tensor(n_eps) for n_eps in self.n_eps]
        all_n_eps = th.stack(all_n_eps).T.to(self.device)
        
        # update the running mean and std of the squared distances
        flattened_squared_distances = th.cat(self.squared_distances, dim=0)
        self.dist_rms.update(flattened_squared_distances)
        self.squared_distances.clear()
        
        # flush the episodic memory of intrinsic rewards
        self.n_eps = [[] for _ in range(self.n_envs)]

        # update the reward module
        self.update(samples)
        
        # scale the intrinsic rewards
        return self.scale(intrinsic_rewards * all_n_eps)

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
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the observations, actions and next observations
        obs_tensor = samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        next_obs_tensor = samples.get("next_observations").to(self.device).view(-1, *self.obs_shape)
        # normalize the observations and next observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs)
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float()
        else:
            actions_tensor = samples.get("actions").view(n_steps * n_envs, -1)
        # create the dataset and loader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        avg_im_loss = []
        avg_fm_loss = []
        # update the encoder, inverse dynamics model and forward dynamics model
        for _idx, batch_data in enumerate(loader):
            # get the batch data
            obs, actions, next_obs = batch_data
            obs, actions, next_obs = obs.to(self.device), actions.to(self.device), next_obs.to(self.device)
            # zero the gradients
            self.encoder_opt.zero_grad()
            self.im_opt.zero_grad()
            self.fm_opt.zero_grad()
            # encode the observations and next observations
            encoded_obs = self.encoder(obs)
            encoded_next_obs = self.encoder(next_obs)
            # get the predicted actions and next observations
            pred_actions = self.im(encoded_obs, encoded_next_obs)
            im_loss = self.im_loss(pred_actions, actions)
            pred_next_obs = self.fm(encoded_obs, actions)
            fm_loss = F.mse_loss(pred_next_obs, encoded_next_obs, reduction="none").mean(dim=-1)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(im_loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # get the masked losses
            im_loss = (im_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            fm_loss = (fm_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            (im_loss + fm_loss).backward()
            self.encoder_opt.step()
            self.im_opt.step()
            self.fm_opt.step()
            avg_im_loss.append(im_loss.item())
            avg_fm_loss.append(fm_loss.item())
        
        self.logger.record("avg_im_loss", np.mean(avg_im_loss))
        self.logger.record("avg_fm_loss", np.mean(avg_fm_loss))