import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hsuanwu.common.typing import *
from hsuanwu.xplore.reward.base import BaseIntrinsicRewardModule


class CnnEncoder(nn.Module):
    """
    Encoder for encoding image-based observations.

    Args:
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.

    Returns:
        CNN-based encoder.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, (8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

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


class MlpEncoder(nn.Module):
    """Encoder for encoding state-based observations.

    Args:
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.

    Returns:
        MLP-based encoder.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.trunk(obs)


class RND(BaseIntrinsicRewardModule):
    """Exploration by Random Network Distillation (RND).
        See paper: https://arxiv.org/pdf/1810.12894.pdf

    Args:
        obs_shape: Data shape of observation.
        action_space: Data shape of action.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        beta: The initial weighting coefficient of the intrinsic rewards.
        kappa: The decay rate.
        latent_dim: The dimension of encoding vectors of the observations.
        lr: The learning rate of inverse and forward dynamics model.
        batch_size: The batch size to train the dynamic models.

    Returns:
        Instance of RND.
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
        lr: float,
        batch_size: int,
    ) -> None:
        super().__init__(obs_shape, action_shape, action_type, device, beta, kappa)

        self._batch_size = batch_size

        if len(self.obs_shape) == 3:
            self.predictor = CnnEncoder(self._obs_shape, latent_dim)
            self.target = CnnEncoder(self._obs_shape, latent_dim)
        else:
            self.predictor = MlpEncoder(self._obs_shape, latent_dim)
            self.target = MlpEncoder(self._obs_shape, latent_dim)

        self.predictor.to(self._device)
        self.target.to(self._device)

        self._opt = optim.Adam(lr=lr, params=self.predictor.parameters())

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False

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
        beta_t = self._beta * np.power(1.0 - self._kappa, step)
        n_steps = rollouts["observations"].shape[0]
        n_envs = rollouts["observations"].shape[1]
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.as_tensor(
            rollouts["observations"], dtype=torch.float32, device=self._device
        )

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.predictor(obs_tensor[:, idx])
                tgt_feats = self.target(obs_tensor[:, idx])
                dist = F.mse_loss(src_feats, tgt_feats, reduction="none").mean(dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
                intrinsic_rewards[:-1, idx, 0] = dist[1:].cpu().numpy()

        # update model
        self.update(rollouts)

        return beta_t * intrinsic_rewards

    def update(
        self,
        rollouts: Dict,
    ) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            rollouts: The collected experiences. A python dict like
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.

        Returns:
            None
        """
        n_steps = rollouts["observations"].shape[0]
        n_envs = rollouts["observations"].shape[1]
        obs_tensor = torch.as_tensor(
            rollouts["observations"], dtype=torch.float32, device=self._device
        ).reshape(n_steps * n_envs, *self._obs_shape)

        dataset = TensorDataset(obs_tensor)
        loader = DataLoader(
            dataset=dataset, batch_size=self._batch_size, drop_last=True
        )

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            src_feats = self.predictor(batch_obs)
            tgt_feats = self.target(batch_obs)

            loss = F.mse_loss(src_feats, tgt_feats)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
