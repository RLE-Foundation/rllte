from torch import nn

import numpy as np
import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.reward.base import BaseIntrinsicRewardModule


class RandomCnnEncoder(nn.Module):
    """
    Random encoder for encoding image-based observations.
    
    Args:
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.
    
    Returns:
        CNN-based random encoder.
    """
    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, (8, 8), stride=(4, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)), nn.ReLU(), nn.Flatten())

        with torch.no_grad():
            sample = torch.ones(size=tuple(obs_shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]
        
        self.linear = nn.Linear(n_flatten, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, obs: Tensor) -> Tensor:
        h = self.trunk(obs)
        h = self.linear(h)
        h = self.layer_norm(h)

        return h


class RandomMlpEncoder(nn.Module):
    """Random encoder for encoding state-based observations.

    Args:
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.
    
    Returns:
        MLP-based random encoder.
    """
    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, obs: Tensor) -> Tensor:
        return self.trunk(obs)


class RISE(BaseIntrinsicRewardModule):
    """Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning (RISE). 
        See paper: https://ieeexplore.ieee.org/abstract/document/9802917/
    
    Args:
        obs_shape: Data shape of observation.
        action_space: Data shape of action.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        beta: The initial weighting coefficient of the intrinsic rewards.
        kappa: The decay rate.
        latent_dim: The dimension of encoding vectors of the observations.
    
    Returns:
        Instance of RISE.
    """
    def __init__(
            self, 
            obs_shape: Tuple,
            action_shape: Tuple,
            action_type: str,
            device: torch.device, 
            beta: float, 
            kappa: float,
            latent_dim: int,
            ) -> None:
        super().__init__(obs_shape, action_shape, action_type, device, beta, kappa)

        if len(self._obs_shape) == 3:
            self.encoder = RandomCnnEncoder(obs_shape=self._obs_shape, latent_dim=latent_dim)
        else:
            self.encoder = RandomMlpEncoder(obs_shape=self._obs_shape, latent_dim=latent_dim)
        
        self.encoder.to(self._device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def compute_irs(self, rollouts: Dict, step: int, alpha: float = 0.5, k: int = 3, average_entropy: bool = False) -> ndarray:
        """Compute the intrinsic rewards using the collected observations.

        Args:
            rollouts: The collected experiences. A python dict like 
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
            step: The current time step.
            alpha: The The order of Rényi entropy.
            k: The k value for marking neighbors.
            average_entropy: Use the average of entropy estimation.

        Returns:
            The intrinsic rewards
        """
        # compute the weighting coefficient of timestep t
        beta_t = self._beta * np.power(1. - self._kappa, step)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        obs_tensor = torch.as_tensor(rollouts['observations'], dtype=torch.float32, device=self._device)


        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                dist = torch.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                if average_entropy:
                    for sub_k in range(k):
                        intrinsic_rewards[:, idx, 0] += torch.pow(
                            torch.kthvalue(dist, sub_k + 1, dim=1).values, 1. - alpha).cpu().numpy()
                    intrinsic_rewards[:, idx, 0] /= k
                else:
                    intrinsic_rewards[:, idx, 0] = torch.pow(
                            torch.kthvalue(dist, k + 1, dim=1).values, 1. - alpha).cpu().numpy()
        
        return beta_t * intrinsic_rewards