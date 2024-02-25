from rlexplore.networks.girm_vae_encoder_decoder import CnnEncoder, CnnDecoder, MlpEncoder, MlpDecoder
from rlexplore.utils.state_process import process

from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np


class VAE(nn.Module):
    def __init__(self,
                 device,
                 model_base,
                 ob_shape,
                 latent_dim,
                 action_dim,
                 ):
        super(VAE, self).__init__()
        if model_base == 'cnn':
            self.encoder = CnnEncoder(
                kwargs={'in_channels': ob_shape[0] * 2}
            )
            self.decoder = CnnDecoder(
                kwargs={'latent_dim': latent_dim, 'action_dim': action_dim, 'out_channels': ob_shape[0]}
            )
        else:
            self.encoder = MlpEncoder(
                kwargs={'input_dim': ob_shape[0] * 2, 'latent_dim': latent_dim}
            )
            self.decoder = MlpDecoder(
                kwargs={'latent_dim': latent_dim, 'action_dim': action_dim, 'obs_dim': ob_shape[0]}
            )

        self.mu = nn.Linear(latent_dim, action_dim)
        self.logvar = nn.Linear(latent_dim, action_dim)

        self.device = device
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def reparameterize(self, mu, logvar, device, training=True):
        # Reparameterization trick as shown in the auto encoding variational bayes paper
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, obs, next_obs):
        latent = self.encoder(obs, next_obs)
        mu = self.mu(latent)
        logvar = self.logvar(latent)

        z = self.reparameterize(mu, logvar, self.device)

        reconstructed_next_obs = self.decoder(z, obs)

        return z, mu, logvar, reconstructed_next_obs


class GIRM(object):
    def __init__(self,
                 envs,
                 device,
                 latent_dim,
                 lr,
                 batch_size,
                 lambd,
                 beta,
                 kappa
                 ):
        """
        Intrinsic Reward Driven Imitation Learning via Generative Model (GIRM)
        Paper: http://proceedings.mlr.press/v119/yu20d/yu20d.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param lr: The learning rate of predictor network.
        :param batch_size: The batch size to train the predictor network.
        :param lambd: The weighting coefficient for combining actions.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """
        if envs.action_space.__class__.__name__ == "Discrete":
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.n
            self.action_type = 'dis'
            self.action_dim = self.action_shape
            self.action_loss = nn.CrossEntropyLoss()
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
            self.action_type = 'cont'
            self.action_dim = self.action_shape[0]
            self.action_loss = nn.MSELoss()
        else:
            raise NotImplementedError
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.lambd = lambd
        self.beta = beta
        self.kappa = kappa

        if len(self.ob_shape) == 3:
            model_base = 'cnn'
            self.vae = VAE(
                device=device,
                model_base=model_base,
                action_dim=self.action_dim,
                ob_shape=self.ob_shape,
                latent_dim=latent_dim
            )
        else:
            model_base = 'mlp'
            self.vae = VAE(
                device=device,
                model_base=model_base,
                action_dim=self.action_dim,
                ob_shape=self.ob_shape,
                latent_dim=latent_dim
            )

        self.vae.to(self.device)
        self.optimizer = optim.Adam(lr=lr, params=self.vae.parameters())

    def get_vae_loss(self, recon_x, x, mean, log_var):
        RECON = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return RECON, KLD

    def compute_irs(self, rollouts, time_steps):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        obs = torch.from_numpy(rollouts['observations'])
        actions = torch.from_numpy(rollouts['actions'])
        obs = obs.to(self.device)
        if self.action_type == 'dis':
            # actions size: (n_steps, n_envs, 1)
            actions = F.one_hot(actions[:, :, 0].to(torch.int64), self.action_shape).float()
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                obs_tensor = obs[:-1, idx]
                actions_tensor = actions[:-1, idx]
                next_obs_tensor = obs[1:, idx]
                # forward prediction
                latent = self.vae.encoder(obs_tensor, next_obs_tensor)
                mu = self.vae.mu(latent)
                logvar = self.vae.logvar(latent)
                z = self.vae.reparameterize(mu, logvar, self.device)
                if self.action_type == 'dis':
                    pred_actions = F.softmax(z, dim=1)
                else:
                    pred_actions = z
                combined_actions = self.lambd * actions_tensor + (1. - self.lambd) * pred_actions
                pred_next_obs = self.vae.decoder(combined_actions, obs_tensor)

                # normalize the observations
                if len(self.ob_shape) == 3:
                    processed_next_obs = process(next_obs_tensor, normalize=True, range=(-1, 1))
                    processed_pred_next_obs = process(pred_next_obs, normalize=True, range=(-1, 1))
                else:
                    processed_next_obs = next_obs_tensor
                    processed_pred_next_obs = pred_next_obs

                intrinsic_rewards[:-1, idx] = F.mse_loss(processed_pred_next_obs, processed_next_obs,
                                                         reduction='mean').cpu().numpy()

        # train the vae model
        self.update(rollouts)

        return beta_t * intrinsic_rewards

    def update(self, rollouts, lambda_recon=1.0, lambda_action=1.0, kld_loss_beta=1.0, lambda_gp=0.0):
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        obs = torch.from_numpy(rollouts['observations']).reshape(n_steps * n_envs, *self.ob_shape)
        if self.action_type == 'dis':
            actions = torch.from_numpy(rollouts['actions']).reshape(n_steps * n_envs, )
            actions = F.one_hot(actions.to(torch.int64), self.action_shape).float()
        else:
            actions = torch.from_numpy(rollouts['actions']).reshape(n_steps * n_envs, self.action_shape[0])
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        # create data loader
        dataset = TensorDataset(obs[:-1], actions[:-1], obs[1:])
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]
            # forward prediction
            latent = self.vae.encoder(batch_obs, batch_next_obs)
            mu = self.vae.mu(latent)
            logvar = self.vae.logvar(latent)
            z = self.vae.reparameterize(mu, logvar, self.device)
            pred_next_obs = self.vae.decoder(z, batch_obs)
            # compute the total loss
            action_loss = self.action_loss(z, batch_actions)
            recon_loss, kld_loss = self.get_vae_loss(pred_next_obs, batch_next_obs, mu, logvar)
            vae_loss = recon_loss + kld_loss_beta * kld_loss + lambda_action * action_loss
            # update
            self.optimizer.zero_grad()
            vae_loss.backward(retain_graph=True)
            self.optimizer.step()
