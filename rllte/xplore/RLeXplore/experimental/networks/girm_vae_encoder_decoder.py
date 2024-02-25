#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：girm_vae_encoder_decoder.py
@Author ：YUAN Mingqi
@Date ：2022/9/21 14:17 
'''

from torch import nn
import torch


class MlpEncoder(nn.Module):
    def __init__(self, kwargs):
        super(MlpEncoder, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(kwargs['input_dim'], 64), nn.LeakyReLU(),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Linear(64, kwargs['latent_dim'])
        )

    def forward(self, obs, next_obs):
        x = torch.cat((obs, next_obs), dim=1)
        return self.main(x)


class MlpDecoder(nn.Module):
    def __init__(self, kwargs):
        super(MlpDecoder, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(kwargs['obs_dim'] + kwargs['action_dim'], 64), nn.LeakyReLU(),
            nn.Linear(64, 64), nn.LeakyReLU(),
            nn.Linear(64, kwargs['obs_dim'])
        )

    def forward(self, z, obs):
        x = torch.cat((z, obs), dim=1)
        return self.main(x)


class CnnEncoder(nn.Module):
    def __init__(self, kwargs):
        super(CnnEncoder, self).__init__()

        self.conv1 = nn.Conv2d(kwargs['in_channels'], 32, kernel_size=3, stride=2)
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
    def __init__(self, kwargs):
        super(CnnDecoder, self).__init__()

        self.linear1 = nn.Linear(kwargs['action_dim'], 64)
        self.linear2 = nn.Linear(64, kwargs['latent_dim'])

        self.conv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
        self.output = nn.ConvTranspose2d(32, out_channels=kwargs['out_channels'], kernel_size=6)
        self.conv9 = nn.Conv2d(kwargs['out_channels'] * 2, out_channels=kwargs['out_channels'],
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
