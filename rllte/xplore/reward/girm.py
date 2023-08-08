# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

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


from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from rllte.common.prototype import BaseIntrinsicRewardModule


class Encoder(nn.Module):
    """Encoder of VAE.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None:
        super().__init__()

        # visual
        if len(obs_shape) == 3:
            self.trunk = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape))
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.linear = nn.Linear(n_flatten, latent_dim)
        else:
            self.trunk = nn.Sequential(nn.Linear(obs_shape[0], 256), nn.ReLU())
            self.linear = nn.Linear(256, latent_dim)

        self.head = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for encoding observations and next-observations.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Encoding vectors.
        """
        h = F.relu(self.linear(self.trunk(obs)))
        next_h = F.relu(self.linear(self.trunk(next_obs)))

        x = self.head(th.cat([h, next_h], dim=1))
        return x

    def encode(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return F.relu(self.linear(self.trunk(obs)))


class Decoder(nn.Module):
    """Decoder of VAE.

    Args:
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Predicted next-observations.
    """

    def __init__(self, action_dim: int, latent_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(action_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: th.Tensor, z: th.Tensor) -> th.Tensor:
        return self.trunk(th.cat([obs, z], dim=1))


class VAE(nn.Module):
    """Variational auto-encoder for reconstructing transition proces.

    Args:
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        VAE instance.
    """

    def __init__(self, device: th.device, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(obs_shape=obs_shape, action_dim=action_dim, latent_dim=latent_dim)
        self.decoder = Decoder(action_dim=action_dim, latent_dim=latent_dim)

        self.mu = nn.Linear(latent_dim, action_dim)
        self.logvar = nn.Linear(latent_dim, action_dim)

        self._device = device
        self.latent_dim = latent_dim

    def reparameterize(self, mu: th.Tensor, logvar: th.Tensor, device: th.device, training: bool = True) -> th.Tensor:
        """Reparameterization trick.

        Args:
            mu (th.Tensor): Mean of the distribution.
            logvar (th.Tensor): Log of the variance of the distribution.
            device (Device): Running device.
            training (bool): True or False.

        Returns:
            Sampled latent vectors.
        """
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        """VAE single forward.
        Args:
            obs (th.Tensor): Observations tensor.
            next_obs (th.Tensor): Next-observations tensor.

        Returns:
            Latent vectors, mean, log of variance, and reconstructed next-observations.
        """
        latent = self.encoder(obs, next_obs)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        z = self.reparameterize(mu, logvar, self._device)

        reconstructed_next_obs = self.decoder(z, obs)

        return z, mu, logvar, reconstructed_next_obs


class GIRM(BaseIntrinsicRewardModule):
    """Intrinsic Reward Driven Imitation Learning via Generative Model (GIRM).
        See paper: http://proceedings.mlr.press/v119/yu20d/yu20d.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for update.
        lambd (float): The weighting coefficient for combining actions.
        lambd_recon (float): Weighting coefficient of the reconstruction loss.
        lambd_action (float): Weighting coefficient of the action loss.
        kld_loss_beta (float): Weighting coefficient of the divergence loss.

    Returns:
        Instance of GIRM.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 0.05,
        kappa: float = 0.000025,
        latent_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 64,
        lambd: float = 0.5,
        lambd_recon: float = 1.0,
        lambd_action: float = 1.0,
        kld_loss_beta: float = 1.0,
    ) -> None:
        super().__init__(observation_space, action_space, device, beta, kappa)

        self.batch_size = batch_size
        self.lambd = lambd
        self.lambd_action = lambd_action
        self.lambd_recon = lambd_recon
        self.kld_loss_beta = kld_loss_beta

        self.vae = VAE(
            device=self._device,
            action_dim=self._action_dim,
            obs_shape=self._obs_shape,
            latent_dim=latent_dim,
        )
        self.vae.to(self._device)

        if self._action_type == "Discrete":
            self.action_loss = nn.CrossEntropyLoss()
        else:
            self.action_loss = nn.MSELoss()

        self.opt = optim.Adam(lr=lr, params=self.vae.parameters())

    def get_vae_loss(self, recon_x: th.Tensor, x: th.Tensor, mean: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        """Compute the vae loss.

        Args:
            recon_x (th.Tensor): Reconstructed x.
            x (th.Tensor): Input x.
            mean (th.Tensor): Sample mean.
            logvar (th.Tensor): Log of the sample variance.

        Returns:
            Loss values.
        """
        RECON = F.mse_loss(recon_x, x)
        KLD = -0.5 * th.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return RECON, KLD

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
        num_steps = samples["obs"].size()[0]
        num_envs = samples["obs"].size()[1]
        obs_tensor = samples["obs"].to(self._device)
        actions_tensor = samples["actions"].to(self._device)
        if self._action_type == "Discrete":
            actions_tensor = F.one_hot(actions_tensor.long(), self._action_dim).float()
            actions_tensor = actions_tensor.to(self._device)
        next_obs_tensor = samples["next_obs"].to(self._device)
        intrinsic_rewards = th.zeros(size=(num_steps, num_envs)).to(self._device)

        with th.no_grad():
            for i in range(num_envs):
                latent = self.vae.encoder(obs_tensor[:, i], next_obs_tensor[:, i])
                mu = self.vae.mu(latent)
                logvar = self.vae.logvar(latent)
                z = self.vae.reparameterize(mu, logvar, self._device)
                if self._action_type == "Discrete":
                    pred_actions = F.softmax(z, dim=1)
                else:
                    pred_actions = z
                combined_actions = self.lambd * actions_tensor[:, i] + (1.0 - self.lambd) * pred_actions
                pred_next_obs = self.vae.decoder(self.vae.encoder.encode(obs_tensor[:, i]), combined_actions)
                intrinsic_rewards[:, i] = F.mse_loss(
                    pred_next_obs,
                    self.vae.encoder.encode(next_obs_tensor[:, i]),
                    reduction="mean",
                )

        # train the vae model
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
        num_steps = samples["obs"].size()[0]
        num_envs = samples["obs"].size()[1]
        obs_tensor = samples["obs"].view((num_envs * num_steps, *self._obs_shape)).to(self._device)
        next_obs_tensor = samples["next_obs"].view((num_envs * num_steps, *self._obs_shape)).to(self._device)

        if self._action_type == "Discrete":
            actions_tensor = samples["actions"].view(num_envs * num_steps).to(self._device)
            actions_tensor = F.one_hot(actions_tensor.long(), self._action_dim).float()
        else:
            actions_tensor = samples["actions"].view((num_envs * num_steps, self._action_dim)).to(self._device)
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size)

        for _idx, batch in enumerate(loader):
            obs, actions, next_obs = batch
            # forward prediction
            latent = self.vae.encoder(obs, next_obs)
            mu = self.vae.mu(latent)
            logvar = self.vae.logvar(latent)
            z = self.vae.reparameterize(mu, logvar, self._device)
            pred_next_obs = self.vae.decoder(self.vae.encoder.encode(obs), z)
            # compute the total loss
            action_loss = self.action_loss(z, actions)
            recon_loss, kld_loss = self.get_vae_loss(pred_next_obs, self.vae.encoder.encode(next_obs), mu, logvar)
            vae_loss = self.lambd_recon * recon_loss + self.kld_loss_beta * kld_loss + self.lambd_action * action_loss
            # update
            self.opt.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.opt.step()
