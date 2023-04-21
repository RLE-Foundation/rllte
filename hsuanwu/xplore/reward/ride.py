from typing import Union, Dict, Tuple
import gymnasium as gym
from omegaconf import DictConfig
from collections import deque

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hsuanwu.xplore.reward.base import BaseIntrinsicRewardModule

class Encoder(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """
    def __init__(self,
                 obs_shape: Tuple,
                 action_shape: Tuple,
                 latent_dim: int
                 ) -> None:
        super().__init__()

        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.Flatten()
            )
            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape))
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.linear = nn.Linear(n_flatten, latent_dim)
        else:
            self.trunk = nn.Sequential(
                nn.Linear(obs_shape[0], 256),
                nn.ReLU()
            )
            self.linear = nn.Linear(256, latent_dim)
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.linear(self.trunk(obs))

class InverseDynamicsModel(nn.Module):
    """Inverse model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """
    def __init__(self, latent_dim, action_dim) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(2 * latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (Tensor): Current observations.
            next_obs (Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        return self.trunk(th.cat([obs, next_obs], dim=1))

class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """
    def __init__(self, latent_dim, action_dim) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, obs: th.Tensor, pred_actions: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs (Tensor): Current observations.
            pred_actions (Tensor): Predicted observations.

        Returns:
            Predicted next-obs.
        """
        return self.trunk(th.cat([obs, pred_actions], dim=1))

class RIDE(BaseIntrinsicRewardModule):
    """RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments.
        See paper: https://arxiv.org/pdf/2002.12292

    Args:
        obs_space (Space or DictConfig): The observation space of environment. When invoked by Hydra, 
            'obs_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like 
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for update.
        capacity (int): The of capacity the episodic memory.
        k (int): Number of neighbors.
        kernel_cluster_distance (float): The kernel cluster distance.
        kernel_epsilon (float): The kernel constant.
        c (float): The pseudo-counts constant.
        sm (float): The kernel maximum similarity.

    Returns:
        Instance of RND.
    """
    def __init__(self, 
                 obs_space: Union[gym.Space, DictConfig],
                 action_space: Union[gym.Space, DictConfig],
                 device: th.device = 'cpu',
                 beta: float = 0.05,
                 kappa: float = 0.000025,
                 latent_dim: int = 128,
                 lr: int = 0.001,
                 batch_size: int = 64,
                 capacity: int = 1000,
                 k: int = 10,
                 kernel_cluster_distance: float = 0.008,
                 kernel_epsilon: float = 0.0001,
                 c: float = 0.001,
                 sm: float = 8.,
    ) -> None:
        super().__init__(obs_space, action_space, device, beta, kappa)
        self.encoder = Encoder(
            obs_shape=obs_space.shape,
            action_shape=action_space.shape,
            latent_dim=latent_dim
        ).to(self._device)

        self.im = InverseDynamicsModel(latent_dim=latent_dim, action_dim=self._action_shape[0]).to(self._device)
        if self._action_shape == "Discrete":
            self.im_loss = nn.CrossEntropyLoss()
        else:
            self.im_loss = nn.MSELoss()

        self.fm = ForwardDynamicsModel(latent_dim=latent_dim, action_dim=self._action_shape[0]).to(self._device)

        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        self.im_opt = th.optim.Adam(self.im.parameters(), lr=lr)
        self.fm_opt = th.optim.Adam(self.fm.parameters(), lr=lr)
        self.batch_size = batch_size

        # episodic memory
        self.episodic_memory = deque(maxlen=capacity)
        self.k = k
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.c = c
        self.sm = sm
    
    def pseudo_counts(self, e: th.Tensor) -> th.Tensor:
        """Pseudo counts.

        Args:
            e (Tensor): Encoded observations.
        
        Returns:
            Conut values.
        """
        num_steps = e.size()[0]
        counts = th.zeros(size=(num_steps, ))
        memory = th.stack(list(self.episodic_memory)).squeeze(1)
        for step in range(num_steps):
            dist = th.norm(e[step] - memory, p=2, dim=1).sort().values[:self.k]
            # moving average
            dist = dist / (dist.mean() + 1e-11)
            dist = th.maximum(dist - self.kernel_cluster_distance, th.zeros_like(dist))
            kernel = self.kernel_epsilon / (dist + self.kernel_epsilon)
            s = th.sqrt(kernel.sum()) + self.c

            if s is th.nan or s > self.sm:
                counts[step] = 0.
            else:
                counts[step] = 1. / s
        
        return counts

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
        num_steps = samples['obs'].size()[0]
        num_envs = samples['obs'].size()[1]
        obs_tensor = samples['obs'].to(self._device)
        actions_tensor = samples['actions']
        if self._action_type == "Discrete":
            actions_tensor = F.one_hot(actions_tensor[:, :, 0].long(), self._action_shape[0]).float()
            actions_tensor = actions_tensor.to(self._device)
        next_obs_tensor = samples['next_obs'].to(self._device)

        intrinsic_rewards = th.zeros(size=(num_steps, num_envs))

        with th.no_grad():
            for i in range(num_envs):
                encoded_obs = self.encoder(obs_tensor[:, i])
                encoded_next_obs = self.encoder(next_obs_tensor[:, i])

                # TODO: add encodings into memory
                self.episodic_memory.extend(encoded_next_obs.split(1))
                n_eps = self.pseudo_counts(e=encoded_next_obs)

                dist = F.mse_loss(encoded_next_obs, encoded_obs, reduction='none').sum(dim=1)
                intrinsic_rewards[:, i] = dist.cpu() * n_eps
        
        self.update(samples)

        return intrinsic_rewards * beta_t
    
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
        num_steps = samples['obs'].size()[0]
        num_envs = samples['obs'].size()[1]
        obs_tensor = samples['obs'].view((num_envs * num_steps, *self._obs_shape)).to(self._device)
        next_obs_tensor = samples['next_obs'].view((num_envs * num_steps, *self._obs_shape)).to(self._device)

        if self._action_type == "Discrete":
            actions_tensor = samples['actions'].view((num_envs * num_steps)).to(self._device)
            actions_tensor = F.one_hot(actions_tensor.long(), self._action_shape[0]).float()
        else:
            actions_tensor = samples['actions'].view((num_envs * num_steps, self._action_shape[0])).to(self._device)

        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size
        )

        for idx, batch in enumerate(loader):
            obs, actions, next_obs = batch

            self.encoder_opt.zero_grad()
            self.im_opt.zero_grad()
            self.fm_opt.zero_grad()

            encoded_obs = self.encoder(obs)
            encoded_next_obs = self.encoder(next_obs)

            pred_actions = self.im(encoded_obs, encoded_next_obs)
            im_loss = self.im_loss(pred_actions, actions)
            pred_next_obs = self.fm(encoded_obs, actions)
            fm_loss = F.mse_loss(pred_next_obs, encoded_next_obs)
            (im_loss + fm_loss).backward()

            self.encoder_opt.step()
            self.im_opt.step()
            self.fm_opt.step()