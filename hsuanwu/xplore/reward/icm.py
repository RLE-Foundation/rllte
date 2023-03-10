from torch.nn import functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch

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
        assert len(obs_shape) >= 3, "CnnEncoder does not support state-based observations! Try image-based observations instead."
        self.trunk = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.LeakyReLU()
        )

        with torch.no_grad():
            n_flatten = self.trunk(torch.as_tensor(np.ones_like(obs_shape)[None]).float()).shape[1]
        
        self.linear = nn.Linear(n_flatten, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, obs: Tensor, next_obs: Tensor) -> Tensor:
        if next_obs is not None:
            input_tensor = torch.cat([obs, next_obs], dim=1)
            h = self.trunk(input_tensor)
            h = self.linear(h)
            h = self.layer_norm(h)
        else:
            h = self.trunk(obs)
            h = self.linear(h)
            h = self.layer_norm(h)

        return h
    


class InverseForwardDynamicsModel(nn.Module):
    """Inverse-Forward model for reconstructing transition process.

    Args:
        latent_dim: The dimension of encoding vectors of the observations.
        action_dim: The dimension of predicted actions.

    Returns:
        Model instance.
    """
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super(InverseForwardDynamicsModel, self).__init__()

        self.inverse_model = nn.Sequential(
            nn.Linear(latent_dim * 2, 64), nn.LeakyReLU(),
            nn.Linear(64, action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.LeakyReLU(),
            nn.Linear(64, latent_dim)
        )

        self.softmax = nn.Softmax()

    def forward(self, obs: Tensor, action: Tensor, next_obs: Tensor, training: bool = True) -> Tensor:
        if training:
            # inverse prediction
            im_input_tensor = torch.cat([obs, next_obs], dim=1)
            pred_action = self.inverse_model(im_input_tensor)
            # forward prediction
            fm_input_tensor = torch.cat([obs, action], dim=-1)
            pred_next_obs = self.forward_model(fm_input_tensor)

            return pred_action, pred_next_obs
        else:
            # forward prediction
            fm_input_tensor = torch.cat([obs, action], dim=-1)
            pred_next_obs = self.forward_model(fm_input_tensor)

            return pred_next_obs



class ICM(BaseIntrinsicRewardModule):
    """Curiosity-Driven Exploration by Self-Supervised Prediction.
        See paper: http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf
    
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
        Instance of ICM.
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
            batch_size: int
            ) -> None:
        super().__init__(obs_shape, action_shape, action_type, device, beta, kappa)

        self._batch_size = batch_size

        if len(self._obs_shape) == 3:
            self.cnn_encoder = CnnEncoder(
                obs_shape=self._obs_shape,
                latent_dim=latent_dim).to(self._device)
            
            self.inverse_forward_model = InverseForwardDynamicsModel(
                latent_dim=latent_dim, 
                action_dim=self._action_shape)
        else:
            # for state-based observations
            self.inverse_forward_model = InverseForwardDynamicsModel(
                latent_dim=self._obs_shape[0], 
                action_dim=self._action_shape)
        
        self._opt = optim.Adam(lr=lr, params=self.inverse_forward_model.parameters())
        self.inverse_forward_model.to(self._device)

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
                if len(self._obs_shape) == 3:
                    encoded_obs = self.cnn_encoder(obs_tensor[:, idx, :, :, :])
                else:
                    encoded_obs = obs_tensor[:, idx]
                pred_next_obs = self.inverse_forward_model(
                    encoded_obs[:-1], actions_tensor[:-1, idx], next_obs=None, training=False)
                processed_next_obs = self._process(encoded_obs[1:], normalize=True, range=(-1, 1))
                processed_pred_next_obs = self._process(pred_next_obs, normalize=True, range=(-1, 1))

                intrinsic_rewards[:-1, idx] = F.mse_loss(processed_pred_next_obs, processed_next_obs, reduction='mean').cpu().numpy()
        
        # update model
        self.update(rollouts)
        
        return beta_t * intrinsic_rewards
    
    def update(self, rollouts: Dict,) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            rollouts: The collected experiences. A python dict like 
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
        
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

        if len(self._obs_shape) == 3:
            encoded_obs = self.cnn_encoder(obs_tensor)
        else:
            encoded_obs = obs_tensor

        dataset = TensorDataset(encoded_obs[:-1], actions_tensor[:-1], encoded_obs[1:])
        loader = DataLoader(dataset=dataset, batch_size=self._batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]

            pred_actions, pred_next_obs = self.inverse_forward_model(
                batch_obs, batch_actions, batch_next_obs
            )

            loss = self.im_loss(pred_actions, batch_actions) + \
                   self.fm_loss(pred_next_obs, batch_next_obs)

            self._opt.zero_grad()
            loss.backward(retain_graph=True)
            self._opt.step()
        