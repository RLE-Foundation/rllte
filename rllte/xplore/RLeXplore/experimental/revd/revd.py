#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines
@File ：revd.py
@Author ：YUAN Mingqi
@Date ：2022/12/03 20:35
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder

import os
import torch
import numpy as np

class REVD(object):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning (REVD)
        Paper: https://openreview.net/pdf?id=V2pw1VYMrDo

        :param obs_shape: The data shape of observations.
        :param action_shape: The data shape of actions.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.beta = beta
        self.kappa = kappa

        if len(self.obs_shape) == 3:
            self.encoder = CnnEncoder(obs_shape, latent_dim)
        else:
            self.encoder = MlpEncoder(obs_shape, latent_dim)

        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.num_updates = 0
        self.last_encoded_obs = list()
    
    def compute_irs(self, rollouts, time_steps, alpha=0.5, k=3, average_divergence=False):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :param alpha: The order of Rényi divergence.
        :param k: The k value.
        :param average_divergence: Use the average of divergence estimation.
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

        if self.num_updates == 0:
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                self.last_encoded_obs.append(src_feats)
            self.num_updates += 1
            return intrinsic_rewards

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                dist_intra = torch.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                dist_outer = torch.linalg.vector_norm(src_feats.unsqueeze(1) - self.last_encoded_obs[idx], ord=2, dim=2)

                if average_divergence:
                    pass
                else:
                    D_step_intra = torch.kthvalue(dist_intra, k + 1, dim=1).values
                    D_step_outer = torch.kthvalue(dist_outer, k + 1, dim=1).values
                    L = torch.kthvalue(dist_intra, 2, dim=1).values.cpu().numpy().sum() / n_steps
                    intrinsic_rewards[:, idx, 0] = L * torch.pow(D_step_outer / (D_step_intra + 0.0001), 1. - alpha).cpu().numpy()

                self.last_encoded_obs[idx] = src_feats
        
        self.num_updates += 1

        return beta_t * intrinsic_rewards