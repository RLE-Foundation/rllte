#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：ride.py
@Author ：YUAN Mingqi
@Date ：2022/12/04 13:47 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder

import torch
import numpy as np

class RIDE:
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments
        Paper: https://arxiv.org/pdf/2002.12292

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

    def pseudo_counts(self,
                     src_feats,
                     k=10,
                     kernel_cluster_distance=0.008,
                     kernel_epsilon=0.0001,
                     c=0.001,
                     sm=8):
        counts = np.zeros(shape=(src_feats.size()[0], ))
        for step in range(src_feats.size()[0]):
            ob_dist = torch.norm(src_feats[step] - src_feats, p=2, dim=1)
            ob_dist = torch.sort(ob_dist).values
            ob_dist = ob_dist[:k]
            dist = ob_dist.cpu().numpy()
            # moving average
            dist = dist / np.mean(dist + 1e-11)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[step] = 0.
            else:
                counts[step] = 1 / s
        return counts
    
    
    def compute_irs(self, rollouts, time_steps):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.from_numpy(rollouts['observations'])
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                dist = torch.linalg.vector_norm(src_feats[:-1] - src_feats[1:], ord=2, dim=1)
                n_eps = self.pseudo_counts(src_feats)
                intrinsic_rewards[:-1, idx, 0] = n_eps[1:] * dist.cpu().numpy()
            
        return beta_t * intrinsic_rewards