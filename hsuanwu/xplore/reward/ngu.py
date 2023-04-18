from typing import Dict, Tuple

import numpy as np
import torch as th
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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

        with th.no_grad():
            sample = th.ones(size=tuple(obs_shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.linear = nn.Linear(n_flatten, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
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

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.trunk(obs)


class NGU(BaseIntrinsicRewardModule):
    """Never Give Up: Learning Directed Exploration Strategies (NGU).
        See paper: https://arxiv.org/pdf/2002.06038

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
        Instance of NGU.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_type: str,
        device: th.device,
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

    def compute_irs(self, rollouts: Dict, step: int) -> np.ndarray:
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
        obs_tensor = th.as_tensor(
            rollouts["observations"], dtype=th.float32, device=self._device
        )

        with th.no_grad():
            for idx in range(n_envs):
                # compute the life-long intrinsic rewards
                rnd_encoded_obs = self.predictor_network(obs_tensor[:, idx])
                rnd_encoded_obs_target = self.target_network(obs_tensor[:, idx])
                dist = th.norm(rnd_encoded_obs - rnd_encoded_obs_target, p=2, dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
                life_long_rewards = dist.cpu().numpy()[1:]
                life_long_rewards = np.where(
                    life_long_rewards >= 1.0, life_long_rewards, 1.0
                )
                # L=5
                life_long_rewards = np.where(
                    life_long_rewards <= 5.0, life_long_rewards, 1.0
                )
                # compute the episodic intrinsic rewards
                if len(self._obs_shape) == 3:
                    encoded_obs = self.target(obs_tensor[:, idx])
                else:
                    encoded_obs = obs_tensor[:, idx]

                episodic_rewards = self.pseudo_counts(encoded_obs)
                intrinsic_rewards[:-1, idx] = episodic_rewards[:-1] * life_long_rewards

        # update the rnd module
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
        obs_tensor = th.as_tensor(
            rollouts["observations"], dtype=th.float32, device=self._device
        ).reshape(n_steps * n_envs, *self._obs_shape)

        dataset = TensorDataset(obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            encoded_obs = self.predictor_network(batch_obs)
            encoded_obs_target = self.target_network(batch_obs)

            loss = F.mse_loss(encoded_obs, encoded_obs_target)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()

    def pseudo_counts(
        self,
        encoded_obs,
        k=10,
        kernel_cluster_distance=0.008,
        kernel_epsilon=0.0001,
        c=0.001,
        sm=8,
    ):
        counts = np.zeros(shape=(encoded_obs.size()[0],))
        for step in range(encoded_obs.size(0)):
            ob_dist = th.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
            ob_dist = th.sort(ob_dist).values
            ob_dist = ob_dist[:k]
            dist = ob_dist.cpu().numpy()
            # moving average
            dist = dist / np.mean(dist)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[step] = 0.0
            else:
                counts[step] = 1 / s
        return counts
