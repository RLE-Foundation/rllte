#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：re3.py.py
@Author ：YUAN Mingqi
@Date ：2022/9/26 18:07 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder

import os
import torch
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    # JAX version of the intrinsic reward function.
    @jax.jit
    def jax_compute_irs(encoded_obs, k):
        dist = jnp.linalg.norm(jnp.expand_dims(encoded_obs, axis=1) - encoded_obs, ord=2, axis=2)
        H_step = jnp.log(jnp.sort(dist, axis=1)[:, k + 1] + 1.)
        return H_step
except:
    pass

class RE3(object):
    def __init__(self,
                 envs,
                 device,
                 enable_jax,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        State Entropy Maximization with Random Encoders for Efficient Exploration (RE3)
        Paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param enable_jax: Use JAX to accelerate the copmutation.
        :param latent_dim: The dimension of encoding vectors of the observations.
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
        self.enable_jax = enable_jax
        self.beta = beta
        self.kappa = kappa

        if len(self.ob_shape) == 3:
            self.encoder = CnnEncoder(
                kwargs={'in_channels': self.ob_shape[0], 'latent_dim': latent_dim})
        else:
            self.encoder = MlpEncoder(
                kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )

        self.encoder.to(self.device)
        if self.enable_jax:
            # allocate GPU memory as needed
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    def compute_irs(self, buffer, time_steps, k=3):
        """
        Compute the intrinsic rewards using the collected observations.
        :param buffer: The experiences buffer.
        :param time_steps: The current time steps.
        :param k: The k value.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = buffer.observations.shape[0]
        n_envs = buffer.observations.shape[1]
        intrinsic_rewards = np.zeros_like(buffer.rewards)

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.from_numpy(buffer.observations)
        obs_tensor = obs_tensor.to(self.device)

        if self.enable_jax:
            for idx in range(n_envs):
                encoded_obs = self.encoder(obs_tensor[:, idx])
                intrinsic_rewards[:, idx] = jax_compute_irs(encoded_obs.cpu().numpy(), k)
        else:
            for idx in range(n_envs):
                encoded_obs = self.encoder(obs_tensor[:, idx])
                dist = torch.norm(encoded_obs.unsqueeze(1) - encoded_obs, p=2, dim=2)
                intrinsic_rewards[:, idx] = torch.log(torch.kthvalue(dist, k + 1, dim=1).values + 1.).cpu().numpy()

        return beta_t * intrinsic_rewards
