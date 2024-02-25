from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class NGU(object):
    def __init__(self,
                 envs,
                 device,
                 latent_dim,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        Never Give Up: Learning Directed Exploration Strategies (NGU)
        Paper: https://arxiv.org/pdf/2002.06038

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param lr: The learning rate of predictor network.
        :param batch_size: The batch size to train the predictor network.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        if envs.action_space.__class__.__name__ == "Discrete":
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.n
        elif envs.action_space.__class__.__name__ == 'Box':
            self.ob_shape = envs.observation_space.shape
            self.action_shape = envs.action_space.shape
        else:
            raise NotImplementedError
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa

        if len(self.ob_shape) == 3:
            self.predictor_network = CnnEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})
            self.target_network = CnnEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})
        else:
            self.predictor_network = MlpEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )
            self.predictor_network = MlpEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )

        if len(self.ob_shape) == 3:
            # use a random network
            self.embedding_network = CnnEncoder(kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})

        self.embedding_network.to(device)
        self.predictor_network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.predictor_network.parameters())

        # freeze the network parameters
        for p in self.target_network.parameters():
            p.requires_grad = False
        for p in self.embedding_network.parameters():
            p.requires_grad = False

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
        obs = obs.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                # compute the life-long intrinsic rewards
                rnd_encoded_obs = self.predictor_network(obs[:, idx])
                rnd_encoded_obs_target = self.target_network(obs[:, idx])
                dist = torch.norm(rnd_encoded_obs - rnd_encoded_obs_target, p=2, dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
                life_long_rewards = dist.cpu().numpy()[1:]
                life_long_rewards = np.where(life_long_rewards >= 1., life_long_rewards, 1.0)
                # L=5
                life_long_rewards = np.where(life_long_rewards <= 5., life_long_rewards, 1.0)
                # compute the episodic intrinsic rewards
                if len(self.ob_shape) == 3:
                    encoded_obs = self.embedding_network(obs[:, idx])
                else:
                    encoded_obs = obs[:, idx]

                episodic_rewards = self.pseudo_counts(encoded_obs)
                intrinsic_rewards[:-1, idx] = episodic_rewards[:-1] * life_long_rewards

        # update the rnd module
        self.update(rollouts)

        return beta_t * intrinsic_rewards

    def update(self, rollouts):
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        obs = torch.from_numpy(rollouts['observations']).reshape(n_steps * n_envs, *self.ob_shape)
        obs = obs.to(self.device)

        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            encoded_obs = self.predictor_network(batch_obs)
            encoded_obs_target = self.target_network(batch_obs)

            loss = F.mse_loss(encoded_obs, encoded_obs_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def pseudo_counts(self,
                     encoded_obs,
                     k=10,
                     kernel_cluster_distance=0.008,
                     kernel_epsilon=0.0001,
                     c=0.001,
                     sm=8,
                     ):
        counts = np.zeros(shape=(encoded_obs.size()[0], ))
        for step in range(encoded_obs.size(0)):
            ob_dist = torch.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
            ob_dist = torch.sort(ob_dist).values
            ob_dist = ob_dist[:k]
            dist = ob_dist.cpu().numpy()
            # TODO: moving average
            dist = dist / np.mean(dist)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[step] = 0.
            else:
                counts[step] = 1 / s
        return counts
