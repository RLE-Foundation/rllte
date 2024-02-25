#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rnd.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 21:46 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class RND(object):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 latent_dim,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        Exploration by Random Network Distillation (RND)
        Paper: https://arxiv.org/pdf/1810.12894.pdf

        :param obs_shape: The data shape of observations.
        :param action_shape: The data shape of actions.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param lr: The learning rate of predictor network.
        :param batch_size: The batch size to train the predictor network.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa

        if len(self.obs_shape) == 3:
            self.predictor = CnnEncoder(obs_shape, latent_dim)
            self.target = CnnEncoder(obs_shape, latent_dim)
        else:
            self.predictor = MlpEncoder(obs_shape, latent_dim)
            self.target = MlpEncoder(obs_shape, latent_dim)

        self.predictor.to(self.device)
        self.target.to(self.device)

        self.opt = optim.Adam(lr=self.lr, params=self.predictor.parameters())

        # freeze the network parameters
        for p in self.target.parameters():
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

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.from_numpy(rollouts['observations'])
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.predictor(obs_tensor[:, idx])
                tgt_feats = self.target(obs_tensor[:, idx])
                dist = F.mse_loss(src_feats, tgt_feats, reduction='none').mean(dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
                intrinsic_rewards[:-1, idx, 0] = dist[1:].cpu().numpy()

        # update the predictor network
        self.update(torch.clone(obs_tensor).reshape(n_steps*n_envs, *obs_tensor.size()[2:]))

        return beta_t * intrinsic_rewards

    def update(self, obs):
        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            src_feats = self.predictor(batch_obs)
            tgt_feats = self.target(batch_obs)

            loss = F.mse_loss(src_feats, tgt_feats)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()