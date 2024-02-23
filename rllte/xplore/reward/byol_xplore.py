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

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from rllte.common.prototype import BaseReward
from .model import ObservationEncoder

th.autograd.set_detect_anomaly(True)

class Byol_Explore(BaseReward):
    """Exploration by Random Network Distillation (RND).
        See paper: https://arxiv.org/pdf/1810.12894.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        n_envs (int): The number of parallel environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        rwd_norm_type (bool): Use running mean and std for reward normalization.
        obs_rms (bool): Use running mean and std for observation normalization.
        gamma (Optional[float]): Intrinsic reward discount rate, None for no discount.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.

    Returns:
        Instance of RND.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        rwd_norm_type: str = "rms",
        obs_rms: bool = True,
        gamma: Optional[float] = None,
        latent_dim: int = 512,
        lstm_latent_dim: int = 256,
        k = 8,
        tau = 0.99,
        lr: float = 0.001,
        batch_size: int = 256,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "default"
    ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, rwd_norm_type, obs_rms, gamma)
        
        # build the online and target networks
        self.encoder = ObservationEncoder(obs_shape=self.obs_shape, 
                                            latent_dim=latent_dim, encoder_model=encoder_model, weight_init=weight_init).to(self.device)
        self.target = ObservationEncoder(obs_shape=self.obs_shape, 
                                         latent_dim=latent_dim, encoder_model=encoder_model, weight_init=weight_init).to(self.device)            
        self.target.load_state_dict(self.encoder.state_dict())

        # build closed loop RNN - it's fed with the latent representation of the observations and the previous action
        self.closed_loop_rnn = th.nn.LSTM(
            input_size=latent_dim + self.policy_action_dim,
            hidden_size=lstm_latent_dim,
            num_layers=1,
        ).to(self.device)

        # build open loop RNN - it's initialized with closed loop representation and predicts future latent representations 
        self.open_loop_rnn = th.nn.LSTM(
            input_size=self.policy_action_dim,
            hidden_size=lstm_latent_dim,
            num_layers=1,
        ).to(self.device)

        # build the encoder predictor - takes the output of the open loop rnn and predicts the next latent representation
        self.encoder_predictor = th.nn.Sequential(
            th.nn.Linear(lstm_latent_dim, latent_dim),
            th.nn.ReLU(),
            th.nn.Linear(latent_dim, latent_dim),
        ).to(self.device)

        self.encoder_opt = th.optim.Adam(list(self.encoder.parameters()) + list(self.encoder_predictor.parameters()), lr=lr)
        self.closed_loop_rnn_opt = th.optim.Adam(self.closed_loop_rnn.parameters(), lr=lr)
        self.open_loop_rnn_opt = th.optim.Adam(self.open_loop_rnn.parameters(), lr=lr)

        # set the parameters
        self.batch_size = batch_size
        self.update_proportion = update_proportion
        self.k = k
        self.tau = tau

        # initialize lstm hidden states
        self.closed_loop_rnn_state = (
            th.zeros(self.closed_loop_rnn.num_layers, self.n_envs, self.closed_loop_rnn.hidden_size).to(device),
            th.zeros(self.closed_loop_rnn.num_layers, self.n_envs, self.closed_loop_rnn.hidden_size).to(device),
        )

        self.open_loop_rnn_state = (
            th.zeros(self.open_loop_rnn.num_layers, self.n_envs, self.open_loop_rnn.hidden_size).to(device),
            th.zeros(self.open_loop_rnn.num_layers, self.n_envs, self.open_loop_rnn.hidden_size).to(device),
        )

    def watch(self, 
              observations: th.Tensor, 
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples, e.g., intrinsic rewards for the current samples. This 
            is useful when applying the memory-based methods to off-policy algorithms.
        """
        
    def compute(self, samples: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors, whose keys are
            'observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'. For example, 
            the data shape of 'observations' is (n_steps, n_envs, *obs_shape). 

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)
        
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the next observations
        obs_tensor = samples.get("observations").to(self.device)
        actions_tensor = samples.get("actions").to(self.device)
        dones_tensor = th.logical_or(samples.get("terminateds"), samples.get("truncateds")).to(self.device) * 1.

        if self.action_type == "Discrete":
            actions_tensor = F.one_hot(actions_tensor.long(), self.policy_action_dim).float().squeeze(2)

        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        
        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs))

        # zero out optimizers
        self.encoder_opt.zero_grad()
        self.closed_loop_rnn_opt.zero_grad()
        self.open_loop_rnn_opt.zero_grad()

        total_loss = 0
        for t in range(n_steps-1):
            d = dones_tensor[t, :]

            # get the closed loop representation
            closed_loop_input = th.cat((self.encoder(obs_tensor[t, :]), actions_tensor[t, :]), dim=1)
            closed_loop_output, self.closed_loop_rnn_state = self.closed_loop_rnn(
                closed_loop_input.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * self.closed_loop_rnn_state[0],
                    (1.0 - d).view(1, -1, 1) * self.closed_loop_rnn_state[1],
                ),
            )

            k_t = min(self.k, n_steps - t - 1)
            first = True
            for k in range(1, k_t):
                d = dones_tensor[t+k, :]

                # get the open loop representation
                open_loop_input = actions_tensor[t+k, :]

                if first:
                    init_hidden = closed_loop_output.clone().detach()
                    open_loop_output, self.open_loop_rnn_state = self.open_loop_rnn(
                        open_loop_input.unsqueeze(0),
                        (
                            (1.0 - d).view(1, -1, 1) * init_hidden,
                            (1.0 - d).view(1, -1, 1) * init_hidden,
                        ),
                    )
                    first = False
                else:
                    open_loop_output, self.open_loop_rnn_state = self.open_loop_rnn(
                        open_loop_input.unsqueeze(0),
                        (
                            (1.0 - d).view(1, -1, 1) * self.open_loop_rnn_state[0],
                            (1.0 - d).view(1, -1, 1) * self.open_loop_rnn_state[1],
                        ),
                    )
                
                # get the predicted representation
                open_loop_output = open_loop_output.squeeze(0)
                predicted_output = self.encoder_predictor(open_loop_output)
                predicted_output = F.normalize(predicted_output, p=2, dim=-1)

                with th.no_grad():
                    # get the target representation
                    target_output = self.encoder(obs_tensor[t+k, :])
                    target_output = F.normalize(target_output, p=2, dim=-1)

                # compute the loss
                loss = F.mse_loss(predicted_output, target_output, reduction="none").mean(dim=-1)

                # a timestep receives intrinsic reward based on how difficult its observation was to predict from past partial histories.
                intrinsic_rewards[t+k, :] += loss.cpu().detach()

                # add loss
                total_loss += loss.sum()

        # average the loss
        total_loss = total_loss / (n_envs * (n_steps - 1))

        # backward
        total_loss.backward()
        self.encoder_opt.step()
        self.closed_loop_rnn_opt.step()
        self.open_loop_rnn_opt.step()

        # update the target network
        self.ema_target()

        # update the reward module
        self.update(samples)
        self.logger.record("avg_byol_loss", total_loss.item())
        
        # scale the intrinsic rewards
        return self.scale(intrinsic_rewards)

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """
        pass

    def ema_target(self) -> None:
        """Update the target network with the exponential moving average."""
        for param, target_param in zip(self.encoder.parameters(), self.target.parameters()):
            target_param.data = target_param.data * self.tau + param.data * (1.0 - self.tau)