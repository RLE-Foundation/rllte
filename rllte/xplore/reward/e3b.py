# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rllte.common.prototype import BaseReward
from .model import DictTensorDataset, InverseDynamicsModel

from rllte.xploit.encoder import MinihackEncoder


class E3B(BaseReward):
    """Exploration via Elliptical Episodic Bonuses (E3B).
        See paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/f4f79698d48bdc1a6dec20583724182b-Paper-Conference.pdf

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

        ridge (float): The ridge parameter for the covariance matrix.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal'].

    Returns:
        Instance of E3B.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "none",
        ridge: float = 0.1,
        latent_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "orthogonal",
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)

        # build the encoder and inverse dynamics model
        self.encoder = MinihackEncoder(
            observation_space=self.observation_space,
        ).to(self.device)
        
        self.im = InverseDynamicsModel(
            latent_dim=latent_dim,
            action_dim=self.policy_action_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)
        # set the loss function
        if self.action_type == "Discrete":
            self.im_loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self.im_loss = nn.MSELoss(reduction="none")
        # set the optimizer
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        self.im_opt = th.optim.Adam(self.im.parameters(), lr=lr)
        # set the parameters
        self.batch_size = batch_size
        self.ridge = ridge
        self.update_proportion = update_proportion
        self.latent_dim = latent_dim
        # set the buffers
        self.cov_inverse = (th.eye(latent_dim) * (1.0 / ridge)).to(self.device)
        self.outer_product_buffer = th.empty(latent_dim, latent_dim).to(self.device)
        self.cov_inverse = self.cov_inverse.repeat(self.n_envs, 1, 1)
        self.outer_product_buffer = self.outer_product_buffer.repeat(self.n_envs, 1, 1)

    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)
        # get the number of steps and environments
        if isinstance(samples.get("next_observations")[0], dict):
            (n_steps, n_envs) = samples.get("next_observations")[0]["glyphs"].size()[:2]
        else:
            (n_steps, n_envs) = samples.get("next_observations").size()[:2]

        # get the observations, terminateds, and truncateds
        if isinstance(samples.get("observations")[0], dict):
            obs_tensor = {
                key: samples.get("observations")[0][key].to(self.device)
                for key in samples.get("observations")[0].keys()
            }
        else:
            obs_tensor = samples.get("observations").to(self.device)
            
        terminateds_tensor = samples.get("terminateds").to(self.device)
        truncateds_tensor = samples.get("truncateds").to(self.device)
        
        # normalize the observations
        if isinstance(obs_tensor, dict):
            for key in obs_tensor.keys():
                obs_tensor[key] = self.normalize(obs_tensor[key], key)
        else:
            obs_tensor = self.normalize(obs_tensor)

        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            for j in range(n_steps):
                if isinstance(obs_tensor, dict):
                    obs_ = {
                        key: obs_tensor[key][j] for key in obs_tensor.keys()
                    }
                else:
                    obs_ = obs_tensor[j]
                
                h = self.encoder(obs_)
                for env_idx in range(n_envs):
                    u = th.mv(self.cov_inverse[env_idx], h[env_idx])
                    b = th.dot(h[env_idx], u).item()
                    intrinsic_rewards[j, env_idx] = b
                    # update the covariance matrix
                    th.outer(u, u, out=self.outer_product_buffer[env_idx])
                    th.add(
                        self.cov_inverse[env_idx],
                        self.outer_product_buffer[env_idx],
                        alpha=-(1.0 / (1.0 + b)),
                        out=self.cov_inverse[env_idx],
                    )
                    # reset the covariance matrix if the episode is terminated or truncated
                    if terminateds_tensor[j, env_idx] or truncateds_tensor[j, env_idx]:
                        self.cov_inverse[env_idx] = th.eye(self.latent_dim) * (
                            1.0 / self.ridge
                        )
        # update the reward module
        if sync:
            self.update(samples)

        # return the scaled intrinsic rewards
        return self.scale(intrinsic_rewards)

    # Usage in your update method
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        # get the number of steps and environments
        if isinstance(samples.get("next_observations")[0], dict):
            (n_steps, n_envs) = samples.get("next_observations")[0]["glyphs"].size()[:2]
        else:
            (n_steps, n_envs) = samples.get("next_observations").size()[:2]

        # get the observations, terminateds, and truncateds
        if isinstance(samples.get("observations")[0], dict):
            obs_tensor = {
                key: samples.get("observations")[0][key].to(self.device).view(-1, *self.obs_shape[key])
                for key in samples.get("observations")[0].keys()
            }
        else:
            obs_tensor = samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        
        if isinstance(samples.get("next_observations")[0], dict):
            next_obs_tensor = {
                key: samples.get("next_observations")[0][key].to(self.device).view(-1, *self.obs_shape[key])
                for key in samples.get("next_observations")[0].keys()
            }
        else:
            next_obs_tensor = samples.get("next_observations").to(self.device).view(-1, *self.obs_shape)

        if isinstance(obs_tensor, dict):
            for key in obs_tensor.keys():
                obs_tensor[key] = self.normalize(obs_tensor[key], key)
                next_obs_tensor[key] = self.normalize(next_obs_tensor[key], key)
        else:
            obs_tensor = self.normalize(obs_tensor)
            next_obs_tensor = self.normalize(next_obs_tensor)

        # process actions
        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs)
            actions_tensor = F.one_hot(
                actions_tensor.long(), self.policy_action_dim
            ).float()
        else:
            actions_tensor = samples.get("actions").view(n_steps * n_envs, -1)

        # create custom dataset
        if isinstance(obs_tensor, dict):
            dataset = DictTensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        else:
            dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)

        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        avg_im_loss = []

        # update the encoder and inverse dynamics model
        for _idx, batch_data in enumerate(loader):
            # get the batch data
            obs, actions, next_obs = batch_data
            
            # Pass the dictionaries directly to the encoder and other models
            obs = {key: value.to(self.device) for key, value in obs.items()}
            next_obs = {key: value.to(self.device) for key, value in next_obs.items()}
            actions = actions.to(self.device)
            
            # Example of using the model
            encoded_obs = self.encoder(obs)
            encoded_next_obs = self.encoder(next_obs)
            pred_actions = self.im(encoded_obs, encoded_next_obs)
            
            # compute the inverse dynamics loss
            im_loss = self.im_loss(pred_actions, actions)
            
            # random mask for subset training
            mask = th.rand(len(im_loss), device=self.device)
            mask = (mask < self.update_proportion).float().to(self.device)
            
            im_loss = (im_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            
            # backward and update
            im_loss.backward()
            self.encoder_opt.step()
            self.im_opt.step()
            
            avg_im_loss.append(im_loss.item())

        # save the average loss
        self.metrics["loss"].append([self.global_step, np.mean(avg_im_loss)])