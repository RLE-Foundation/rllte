#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：inverse_forward_networks.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 14:07 
'''

from torch import nn, optim
import torch

class InverseForwardDynamicsModel(nn.Module):
    def __init__(self, kwargs):
        super(InverseForwardDynamicsModel, self).__init__()

        self.inverse_model = nn.Sequential(
            nn.Linear(kwargs['latent_dim'] * 2, 64), nn.LeakyReLU(),
            nn.Linear(64, kwargs['action_dim'])
        )

        self.forward_model = nn.Sequential(
            nn.Linear(kwargs['latent_dim'] + kwargs['action_dim'], 64), nn.LeakyReLU(),
            nn.Linear(64, kwargs['latent_dim'])
        )

        self.softmax = nn.Softmax()

    def forward(self, obs, action, next_obs, training=True):
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

class CnnEncoder(nn.Module):
    def __init__(self, kwargs):
        super(CnnEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(kwargs['in_channels'], 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64), nn.LeakyReLU()
        )

    def forward(self, obs, next_obs=None):
        if next_obs is not None:
            input_tensor = torch.cat([obs, next_obs], dim=1)
        else:
            input_tensor = obs

        latent_vectors = self.main(input_tensor)

        return latent_vectors.view(latent_vectors.size(0), -1)
