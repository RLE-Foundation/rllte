from typing import Union, Dict, Tuple
import gymnasium as gym
from omegaconf import DictConfig

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

class RND(BaseIntrinsicRewardModule):
    """Exploration by Random Network Distillation (RND).
        See paper: https://arxiv.org/pdf/1810.12894.pdf

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
                 batch_size: int = 64
    ) -> None:
        super().__init__(obs_space, action_space, device, beta, kappa)
        self.predictor = Encoder(
            obs_shape=obs_space.shape,
            action_shape=action_space.shape,
            latent_dim=latent_dim
        ).to(self._device)

        self.target = Encoder(
            obs_shape=obs_space.shape,
            action_shape=action_space.shape,
            latent_dim=latent_dim
        ).to(self._device)

        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.batch_size = batch_size

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False
    
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
        next_obs_tensor = samples['next_obs'].to(self._device)

        intrinsic_rewards = th.zeros(size=(num_steps, num_envs))

        with th.no_grad():
            for i in range(num_envs):
                src_feats = self.predictor(next_obs_tensor[:, i])
                tgt_feats = self.target(next_obs_tensor[:, i])
                dist = F.mse_loss(src_feats, tgt_feats, reduction='none').mean(dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
                intrinsic_rewards[:, i] = dist
        
        # udpate the module
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

        dataset = TensorDataset(obs_tensor)
        loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size
        )

        for idx, batch_data in enumerate(loader):
            obs = batch_data[0]
            src_feats = self.predictor(obs)
            with th.no_grad():
                tgt_feats = self.target(obs)

            self.opt.zero_grad()
            loss = F.mse_loss(src_feats, tgt_feats)
            loss.backward()
            self.opt.step()
