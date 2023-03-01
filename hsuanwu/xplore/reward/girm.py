from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch

from hsuanwu.common.typing import *
from hsuanwu.xplore.reward.base import BaseRewardIntrinsicModule


class MlpEncoder(nn.Module):
    """MLP-based encoder of VAE.

    Args:
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.
    
    Returns:
        MLP-based encoder.
    """
    def __init__(self, obs_shape: Tuple, latent_dim: int) -> None:
        super(MlpEncoder, self).__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], 64), nn.LeakyReLU(),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        x = torch.cat((obs, next_obs), dim=1)
        return self.trunk(x)


class MlpDecoder(nn.Module):
    """MLP-based decoder of VAE.

    Args:
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.
    
    Returns:
        MLP-based decoder.
    """
    def __init__(self, obs_shape: Tuple, action_dim: int) -> None:
        super(MlpDecoder, self).__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0] + action_dim, 64), nn.LeakyReLU(),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Linear(64, obs_shape[0])
        )

    def forward(self, z: Tensor, obs: Tensor) -> Tensor:
        x = torch.cat((z, obs), dim=1)
        return self.trunk(x)


class CnnEncoder(nn.Module):
    """CNN-based encoder of VAE.
    
    Args:
        obs_shape: The data shape of observations.
    
    Returns:
        CNN-based encoder.
    """
    def __init__(self, obs_shape: Tuple) -> None:
        super(CnnEncoder, self).__init__()

        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.lrelu = nn.LeakyReLU()

        # Initialize the weights using xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, obs, next_obs):
        x = torch.cat((obs, next_obs), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = x.view(x.size(0), -1)

        return x


class CnnDecoder(nn.Module):
    """CNN-based decoder of VAE.
    
    Args:
        obs_shape: The data shape of observations.
    
    Returns:
        CNN-based decoder.
    """
    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None:
        super(CnnDecoder, self).__init__()

        self.linear1 = nn.Linear(action_dim, 64)
        self.linear2 = nn.Linear(64, latent_dim)

        self.conv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        self.output = nn.ConvTranspose2d(32, out_channels=obs_shape[0], kernel_size=6)
        self.conv9 = nn.Conv2d(obs_shape[0] * 2, out_channels=obs_shape[0],
                               kernel_size=3, stride=1, dilation=2, padding=2)
        self.lrelu = nn.LeakyReLU()

        # Initialize weights using xavier initialization
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.conv9.weight)

    def forward(self, z, obs):
        batch_size, _ = z.size()
        z = self.linear1(z)
        z = self.lrelu(z)
        z = self.linear2(z)
        z = self.lrelu(z)
        z = z.view((batch_size, 64, 4, 4))

        z = self.conv5(z)
        z = self.lrelu(z)

        z = self.conv6(z)
        z = self.lrelu(z)

        z = self.conv7(z)
        z = self.lrelu(z)

        z = self.conv8(z)
        z = self.lrelu(z)

        z = self.output(z)

        z = torch.cat((z, obs), dim=1)
        output = self.conv9(z)

        return output


class VAE(nn.Module):
    """Variational auto-encoder for reconstructing transition proces.
    
    Args:
        device: Device (cpu, cuda, ...) on which the code should be run.
        obs_shape: The data shape of observations.
        latent_dim: The dimension of encoding vectors of the observations.
        action_dim: The dimension of predicted actions.

    """
    def __init__(self,
                 device: torch.device,
                 obs_shape: Tuple,
                 latent_dim: int,
                 action_dim: int) -> None:
        super(VAE, self).__init__()
        if len(obs_shape) == 3:
            self.encoder = CnnEncoder(obs_shape=obs_shape)
            self.decoder = CnnDecoder(obs_shape=obs_shape, action_dim=action_dim, latent_dim=latent_dim)
        else:
            self.encoder = MlpEncoder(obs_shape=obs_shape, latent_dim=latent_dim)
            self.decoder = MlpDecoder(obs_shape=obs_shape, action_dim=action_dim)

        self.mu = nn.Linear(latent_dim, action_dim)
        self.logvar = nn.Linear(latent_dim, action_dim)

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def reparameterize(self, mu: Tensor, logvar: Tensor, device: torch.device, training: bool = True) -> Tensor:
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        latent = self.encoder(obs, next_obs)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        z = self.reparameterize(mu, logvar, self.device)

        reconstructed_next_obs = self.decoder(z, obs)

        return z, mu, logvar, reconstructed_next_obs


class GIRM(BaseRewardIntrinsicModule):
    """Intrinsic Reward Driven Imitation Learning via Generative Model (GIRM).
        See paper: http://proceedings.mlr.press/v119/yu20d/yu20d.pdf
    
    Args:
        env: The environment.
        device: Device (cpu, cuda, ...) on which the code should be run.
        beta: The initial weighting coefficient of the intrinsic rewards.
        kappa: The decay rate.
        latent_dim: The dimension of encoding vectors of the observations.
        lr: The learning rate of inverse and forward dynamics model.
        batch_size: The batch size to train the dynamic models.
        lambd: The weighting coefficient for combining actions.
    
    Returns:
        Instance of GIRM.
    """
    def __init__(
            self, 
            env: Env, 
            device: torch.device, 
            beta: float, 
            kappa: float,
            latent_dim: int,
            lr: float,
            batch_size: int,
            lambd: float
            ) -> None:
        super().__init__(env, device, beta, kappa)

        self._batch_size = batch_size
        self._lambd = lambd

        self._vae = VAE(
            device=self._device,
            action_dim=self._action_shape,
            obs_shape=self._obs_shape,
            latent_dim=latent_dim)
        self._vae.to(self._device)

        if self._action_type == 'dis':
            self._action_loss = nn.CrossEntropyLoss()
        else:
            self._action_loss = nn.MSELoss()

        self._opt = optim.Adam(lr=lr, params=self._vae.parameters())
    
    def _get_vae_loss(self, recon_x: Tensor, x: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
        RECON = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return RECON, KLD


    def _process(self, tensor: Tensor, range: Tuple) -> Tensor:
        """Make a grid of images.

         Args:
            tensor: 4D mini-batch Tensor of shape (B x C x H x W)
                or a list of images all of the same size.
            range: tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
        
        Returns:
            Processed tensors.
        """
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

            def norm_ip(img, min, max):
                img.clamp_(min=min, max=max)
                img.add_(-min).div_(max - min + 1e-5)

            def norm_range(t, range):
                if range is not None:
                    norm_ip(t, range[0], range[1])
                else:
                    norm_ip(t, float(t.min()), float(t.max()))
        
        norm_range(tensor, range)
        
        return tensor

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
        actions_tensor = torch.from_numpy(rollouts['actions'])
        if self.action_type == 'dis':
            # actions size: (n_steps, n_envs, 1)
            actions_tensor = F.one_hot(actions_tensor[:, :, 0].to(torch.int64), self._action_shape).float()
        obs_tensor = obs_tensor.to(self.device)
        actions_tensor = actions_tensor.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                obs_tensor = obs_tensor[:-1, idx]
                actions_tensor = actions_tensor[:-1, idx]
                next_obs_tensor = obs_tensor[1:, idx]
                # forward prediction
                latent = self._vae.encoder(obs_tensor, next_obs_tensor)
                mu = self._vae.mu(latent)
                logvar = self.vae.logvar(latent)
                z = self._vae.reparameterize(mu, logvar, self.device)
                if self.action_type == 'dis':
                    pred_actions = F.softmax(z, dim=1)
                else:
                    pred_actions = z
                combined_actions = self.lambd * actions_tensor + (1. - self.lambd) * pred_actions
                pred_next_obs = self._vae.decoder(combined_actions, obs_tensor)

                # normalize the observations
                if len(self.ob_shape) == 3:
                    processed_next_obs = self._process(next_obs_tensor, normalize=True, range=(-1, 1))
                    processed_pred_next_obs = self._process(pred_next_obs, normalize=True, range=(-1, 1))
                else:
                    processed_next_obs = next_obs_tensor
                    processed_pred_next_obs = pred_next_obs

                intrinsic_rewards[:-1, idx] = F.mse_loss(processed_pred_next_obs, processed_next_obs,
                                                         reduction='mean').cpu().numpy()

        # train the vae model
        self.update(rollouts)

        return beta_t * intrinsic_rewards
    
    def update(self, 
               rollouts: Dict, 
               lambda_recon: float = 1.0, 
               lambda_action: float = 1.0, 
               kld_loss_beta: float = 1.0) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            rollouts: The collected experiences. A python dict like 
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
            lambda_recon: Weighting coefficient of the reconstruction loss.
            lambda_action: Weighting coefficient of the action loss.
            kld_loss_beta: Weighting coefficient of the divergence loss.
        
        Returns:
            None
        """
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        obs_tensor = torch.from_numpy(rollouts['observations']).reshape(n_steps * n_envs, *self._obs_shape)
        if self.action_type == 'dis':
            actions_tensor = torch.from_numpy(rollouts['actions']).reshape(n_steps * n_envs, )
            actions_tensor = F.one_hot(actions_tensor.to(torch.int64), self._action_shape).float()
        else:
            actions_tensor = torch.from_numpy(rollouts['actions']).reshape(n_steps * n_envs, self._action_shape)
        obs_tensor = obs_tensor.to(self.device)
        actions_tensor = actions_tensor.to(self.device)

        obs_tensor = obs_tensor.to(self.device)
        actions_tensor = actions_tensor.to(self.device)
        # create data loader
        dataset = TensorDataset(obs_tensor[:-1], actions_tensor[:-1], obs_tensor[1:])
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]
            # forward prediction
            latent = self._vae.encoder(batch_obs, batch_next_obs)
            mu = self._vae.mu(latent)
            logvar = self._vae.logvar(latent)
            z = self._vae.reparameterize(mu, logvar, self.device)
            pred_next_obs = self._vae.decoder(z, batch_obs)
            # compute the total loss
            action_loss = self._action_loss(z, batch_actions)
            recon_loss, kld_loss = self._get_vae_loss(pred_next_obs, batch_next_obs, mu, logvar)
            vae_loss = lambda_recon *  recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss
            # update
            self._opt.zero_grad()
            vae_loss.backward(retain_graph=True)
            self._opt.step()