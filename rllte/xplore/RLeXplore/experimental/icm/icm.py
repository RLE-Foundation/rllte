from rlexplore.networks.inverse_forward_networks import InverseForwardDynamicsModel, CnnEncoder

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

class ICM(object):
    def __init__(self,
                 envs,
                 device,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        Curiosity-Driven Exploration by Self-Supervised Prediction
        Paper: http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param lr: The learning rate of inverse and forward dynamics model.
        :param batch_size: The batch size to train the dynamics model.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """
        self.device = device
        self.beta = beta
        self.kappa = kappa
        self.lr = lr
        self.batch_size = batch_size

        if envs.action_space.__class__.__name__ == "Discrete":
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.n
            self.action_type = 'dis'
            self.inverse_forward_model = InverseForwardDynamicsModel(
                kwargs={'latent_dim': 1024, 'action_dim': self.action_shape}
            ).to(device)
            self.im_loss = nn.CrossEntropyLoss()
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
            self.action_type = 'cont'
            self.inverse_forward_model = InverseForwardDynamicsModel(
                kwargs={'latent_dim': self.ob_shape[0], 'action_dim': self.action_shape[0]}
            ).to(device)
            self.im_loss = nn.MSELoss()
        else:
            raise NotImplementedError
        self.fm_loss = nn.MSELoss()

        if len(self.ob_shape) == 3:
            self.cnn_encoder = CnnEncoder(kwargs={'in_channels': 4}).to(device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.inverse_forward_model.parameters())

    def update(self, rollouts):
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

        if len(self.ob_shape) == 3:
            encoded_obs = self.cnn_encoder(obs)
        else:
            encoded_obs = obs

        dataset = TensorDataset(encoded_obs[:-1], actions[:-1], encoded_obs[1:])
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]

            pred_actions, pred_next_obs = self.inverse_forward_model(
                batch_obs, batch_actions, batch_next_obs
            )

            loss = self.im_loss(pred_actions, batch_actions) + \
                   self.fm_loss(pred_next_obs, batch_next_obs)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

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
        if self.action_type == 'dis':
            # actions size: (n_steps, n_envs, 1)
            actions = F.one_hot(actions[:, :, 0].to(torch.int64), self.action_shape).float()
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                if len(self.ob_shape) == 3:
                    encoded_obs = self.cnn_encoder(obs[:, idx, :, :, :])
                else:
                    encoded_obs = obs[:, idx]
                pred_next_obs = self.inverse_forward_model(
                    encoded_obs[:-1], actions[:-1, idx], next_obs=None, training=False)
                processed_next_obs = torch.clip(encoded_obs[1:], min=-1.0, max=1.0)
                processed_pred_next_obs = torch.clip(pred_next_obs, min=-1.0, max=1.0)

                intrinsic_rewards[:-1, idx] = F.mse_loss(processed_pred_next_obs, processed_next_obs, reduction='mean').cpu().numpy()
            # processed_next_obs = process(encoded_obs[1:n_steps], normalize=True, range=(-1, 1))
            # processed_pred_next_obs = process(pred_next_obs, normalize=True, range=(-1, 1))
        # train the icm
        self.update(rollouts)

        return beta_t * intrinsic_rewards
