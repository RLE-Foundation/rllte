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
            n_flatten = self.trunk(torch.as_tensor(np.ones_like(obs_shape)[None]).float()).shape[1]
        
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


class RIDE(BaseIntrinsicRewardModule):
    """RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments.
        See paper: https://arxiv.org/pdf/2002.12292
    
    Args:
        obs_shape: Data shape of observation.
        action_space: Data shape of action.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        beta: The initial weighting coefficient of the intrinsic rewards.
        kappa: The decay rate.
        latent_dim: The dimension of encoding vectors of the observations.
    
    Returns:
        Instance of RIDE.
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
    
    def pseudo_counts(self,
                     src_feats,
                     k=10,
                     kernel_cluster_distance=0.008,
                     kernel_epsilon=0.0001,
                     c=0.001,
                     sm=8):
        counts = np.zeros(shape=(src_feats.size()[0], ))
        for step in range(src_feats.size()[0]):
            ob_dist = torch.norm(src_feats[step] - src_feats, p=2, dim=1)
            ob_dist = torch.sort(ob_dist).values
            ob_dist = ob_dist[:k]
            dist = ob_dist.cpu().numpy()
            # moving average
            dist = dist / np.mean(dist + 1e-11)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[step] = 0.
            else:
                counts[step] = 1 / s
        return 
    
    def compute_irs(self, rollouts: Dict, step: int) -> ndarray:
        """Compute the intrinsic rewards using the collected observations.

        Args:
            rollouts: The collected experiences. A python dict like 
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
            step: The current time step.

        Returns:
            The intrinsic rewards
        """
        # compute the weighting coefficient of timestep t
        beta_t = self._beta * np.power(1. - self._kappa, step)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        obs_tensor = torch.from_numpy(rollouts['observations'])
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                dist = torch.linalg.vector_norm(src_feats[:-1] - src_feats[1:], ord=2, dim=1)
                n_eps = self.pseudo_counts(src_feats)
                intrinsic_rewards[:-1, idx, 0] = n_eps[1:] * dist.cpu().numpy()
            
        return beta_t * intrinsic_rewards