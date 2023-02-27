from torch import nn

import numpy as np
import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.reward.base import BaseRewardModule


class RandomCnnEncoder(nn.Module):
    """
    Random encoder for encoding image-based observations.
    
    :param obs_shape: The data shape of observations.
    :param latent_dim: The dimension of encoding vectors of the observations.
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

    def forward(self, obs) -> Tensor:
        h = self.trunk(obs)
        h = self.linear(h)
        h = self.layer_norm(h)

        return h


class RandomMlpEncoder(nn.Module):
    """
    Random encoder for encoding state-based observations.
    
    :param obs_shape: The data shape of observations.
    :param latent_dim: The dimension of encoding vectors of the observations.
    """
    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, obs) -> Tensor:
        return self.trunk(obs)


class RE3(BaseRewardModule):
    """
    State Entropy Maximization with Random Encoders for Efficient Exploration (RE3)
    Paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf
    
    :param env: The environment.
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    :param beta: The initial weighting coefficient of the intrinsic rewards.
    :param kappa: The decay rate.
    :param latent_dim: The dimension of encoding vectors of the observations.
    """
    def __init__(
            self, 
            env: Env, 
            device: torch.device, 
            beta: float, 
            kappa: float,
            latent_dim: int,
            ) -> None:
        super().__init__(env, device, beta, kappa)

        if len(self._obs_shape) == 3:
            self.encoder = RandomCnnEncoder(obs_shape=self._obs_shape, latent_dim=latent_dim)
        else:
            self.encoder = RandomMlpEncoder(obs_shape=self._obs_shape, latent_dim=latent_dim)
        
        self.encoder.to(self._device)

    # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def compute_irs(self, rollouts: Dict, step: int, k: int = 3, average_entropy: bool = False) -> ndarray:
        """
        Compute the intrinsic rewards using the collected observations.

        :param rollouts: The collected experiences. A python dict like:
            + observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>
            - actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>
            + rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>
        :param step: The current time step.
        :param k: The k value for marking neighbors.
        :param average_entropy: Use the average of entropy estimation.

        :return: The intrinsic rewards
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
                dist = torch.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                if average_entropy:
                    for sub_k in range(k):
                        intrinsic_rewards[:, idx, 0] += torch.log(
                            torch.kthvalue(dist, sub_k + 1, dim=1).values + 1.).cpu().numpy()
                    intrinsic_rewards[:, idx, 0] /= k
                else:
                    intrinsic_rewards[:, idx, 0] = torch.log(
                            torch.kthvalue(dist, k + 1, dim=1).values + 1.).cpu().numpy()
        
        return beta_t * intrinsic_rewards