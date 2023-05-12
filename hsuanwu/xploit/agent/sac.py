import os
from pathlib import Path
from typing import Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from omegaconf import DictConfig
from torch.nn import functional as F

from hsuanwu.xploit.agent import utils
from hsuanwu.xploit.agent.base import BaseAgent
from hsuanwu.xploit.agent.networks import (OffPolicyDoubleCritic, 
                                           OffPolicyStochasticActor,
                                           get_network_init,
                                           ExportModel)


class SAC(BaseAgent):
    """Soft Actor-Critic (SAC) agent.
        When 'augmentation' module is invoked, this learner will transform into Data Regularized Q (DrQ) agent.
        Based on: https://github.com/denisyarats/pytorch_sac

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": action_space.shape, "n": action_space.n, "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        hidden_dim (int): The size of the hidden layers.
        critic_target_tau (float): The critic Q-function soft-update rate.
        update_every_steps (int): The agent update frequency.
        log_std_range (Tuple[float]): Range of std for sampling actions.
        betas (Tuple[float]): coefficients used for computing running averages of gradient and its square.
        temperature (float): Initial temperature coefficient.
        fixed_temperature (bool): Fixed temperature or not.
        discount (float): Discount factor.
        network_init_method (str): Network initialization method name.

    Returns:
        Soft Actor-Critic learner instance.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str,
        feature_dim: int,
        lr: float = 1e-4,
        eps: float = 1e-8,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.005,
        update_every_steps: int = 2,
        log_std_range: Tuple[float] = (-5.0, 2),
        betas: Tuple[float] = (0.9, 0.999),
        temperature: float = 0.1,
        fixed_temperature: bool = False,
        discount: float = 0.99,
        network_init_method: str = "orthogonal"
    ) -> None:
        super().__init__(observation_space, action_space, device, feature_dim, lr, eps)

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.fixed_temperature = fixed_temperature
        self.discount = discount
        self.betas = betas
        self.network_init_method = network_init_method

        # create models
        self.actor = OffPolicyStochasticActor(
            action_dim=self.action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            log_std_range=log_std_range,
        ).to(self.device)

        self.critic = OffPolicyDoubleCritic(
            action_dim=self.action_dim, 
            feature_dim=feature_dim, 
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic_target = OffPolicyDoubleCritic(
            action_dim=self.action_dim, 
            feature_dim=feature_dim, 
            hidden_dim=hidden_dim,
        ).to(self.device)

        # target entropy
        self.target_entropy = - self.action_dim
        self.log_alpha = th.tensor(np.log(temperature), 
                                   device=self.device, 
                                   requires_grad=True)

    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)

    def integrate(self, **kwargs) -> None:
        """Integrate agent and other modules (encoder, reward, ...) together"""
        # set encoder and distribution
        self.encoder = kwargs["encoder"]
        self.dist = kwargs["dist"]
        self.actor.dist = kwargs["dist"]
        # network initialization
        self.encoder.apply(get_network_init(self.network_init_method))
        self.actor.apply(get_network_init(self.network_init_method))
        self.critic.apply(get_network_init(self.network_init_method))
        # synchronize critic and target critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        # create optimizers
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=self.lr, eps=self.eps)
        self.actor_opt = th.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.eps)
        self.critic_opt = th.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)
        self.log_alpha_opt = th.optim.Adam([self.log_alpha], lr=self.lr, betas=self.betas)
        # set training mode
        self.train()
        # set augmentation and intrinsic reward
        if kwargs["aug"] is not None:
            self.aug = kwargs["aug"]
        if kwargs["irs"] is not None:
            self.irs = kwargs["irs"]

    @property
    def alpha(self) -> th.Tensor:
        """Get the temperature coefficient."""
        return self.log_alpha.exp()

    def act(self, obs: th.Tensor, training: bool = True, step: int = 0) -> Tuple[th.Tensor]:
        """Sample actions based on observations.

        Args:
            obs (Tensor): Observations.
            training (bool): training mode, True or False.
            step (int): Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        dist = self.actor.get_dist(obs=encoded_obs, step=step)

        if not training:
            action = dist.mean
        else:
            action = dist.sample()

        return action

    def update(self, replay_storage, step: int = 0) -> Dict[str, float]:
        """Update the learner.

        Args:
            replay_storage (Storage): Hsuanwu replay storage.
            step (int): Global training step.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """
        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        # weights for PrioritizedReplayStorage
        (
            indices,
            obs,
            action,
            reward,
            terminated,
            next_obs,
            weights,
        ) = replay_storage.sample(step)

        if self.irs is not None:
            intrinsic_reward = self.irs.compute_irs(
                samples={
                    "obs": obs.unsqueeze(1),
                    "actions": action.unsqueeze(1),
                    "next_obs": next_obs.unsqueeze(1),
                },
                step=step,
            )
            reward += intrinsic_reward.to(self.device)

        # obs augmentation
        if self.aug is not None:
            aug_obs = self.aug(obs.clone().float())
            aug_next_obs = self.aug(next_obs.clone().float())
            assert (
                aug_obs.size() == obs.size() and aug_obs.dtype == obs.dtype
            ), "The data shape and data type should be consistent after augmentation!"
            with th.no_grad():
                encoded_aug_obs = self.encoder(aug_obs)
            encoded_aug_next_obs = self.encoder(aug_next_obs)
        else:
            encoded_aug_obs = None
            encoded_aug_next_obs = None

        # encode
        encoded_obs = self.encoder(obs)
        with th.no_grad():
            encoded_next_obs = self.encoder(next_obs)

        # update criitc
        metrics.update(
            self.update_critic(
                obs=encoded_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                next_obs=encoded_next_obs,
                weights=weights,
                aug_obs=encoded_aug_obs,
                aug_next_obs=encoded_aug_next_obs,
                step=step,
            )
        )

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor_and_alpha(encoded_obs.detach(), weights, step))

        # udpate critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        metrics.update({"indices": indices})

        return metrics

    def update_critic(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        terminated: th.Tensor,
        next_obs: th.Tensor,
        weights: th.Tensor,
        aug_obs: th.Tensor,
        aug_next_obs: th.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """Update the critic network.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.
            reward (Tensor): Rewards.
            terminated (Tensor): Terminateds.
            next_obs (Tensor): Next observations.
            weights (Tensor): Batch sample weights.
            aug_obs (Tensor): Augmented observations.
            aug_next_obs (Tensor): Augmented next observations.
            step (int): Global training step.

        Returns:
            Critic loss metrics.
        """
        with th.no_grad():
            dist = self.actor.get_dist(next_obs, step=step)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = th.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (1.0 - terminated) * self.discount * target_V

            # enable observation augmentation
            if self.aug is not None:
                dist_aug = self.actor.get_dist(aug_next_obs, step=step)
                next_action_aug = dist_aug.rsample()
                log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
                target_Q1, target_Q2 = self.critic_target(aug_next_obs, next_action_aug)
                target_V = th.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
                target_Q_aug = reward + (1.0 - terminated) * self.discount * target_V
                # mixed target Q-function
                target_Q = (target_Q + target_Q_aug) / 2

        Q1, Q2 = self.critic(obs, action)
        TDE1 = target_Q - Q1
        TDE2 = target_Q - Q2
        critic_loss = (0.5 * weights * (TDE1.pow(2) + TDE2.pow(2))).mean()
        # TODO: for PrioritizedReplayStorage
        priorities = abs(((TDE1 + TDE2) / 2.0 + 1e-5).squeeze())
        # critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.aug is not None:
            Q1_aug, Q2_aug = self.critic(aug_obs, action)
            critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(Q2_aug, target_Q)

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "critic_q1": Q1.mean().item(),
            "critic_q2": Q2.mean().item(),
            "critic_target": target_Q.mean().item(),
            "priorities": priorities.data.cpu().numpy(),
        }

    def update_actor_and_alpha(self, obs: th.Tensor, weights: th.Tensor, step: int) -> Dict[str, float]:
        """Update the actor network and temperature.

        Args:
            obs (Tensor): Observations.
            weights (Tensor): Batch sample weights.
            step (int): Global training step.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self.actor.get_dist(obs, step=step)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = th.min(Q1, Q2)

        actor_loss = ((self.alpha.detach() * log_prob - Q) * weights).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if not self.fixed_temperature:
            # update temperature
            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach() * weights).mean()
            alpha_loss.backward()
            self.log_alpha_opt.step()
        else:
            alpha_loss = th.scalar_tensor(s=0.0)

        return {"actor_loss": actor_loss.item(), "alpha_loss": alpha_loss.item()}

    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Storage path.

        Returns:
            None.
        """
        if "pretrained" in str(path):  # pretraining
            th.save(self.encoder.state_dict(), path / "encoder.pth")
            th.save(self.actor.state_dict(), path / "actor.pth")
            th.save(self.critic.state_dict(), path / "critic.pth")
        else:
            export_model = ExportModel(encoder=self.encoder, actor=self.actor)
            th.save(export_model, path / "agent.pth")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        encoder_params = th.load(os.path.join(path, "encoder.pth"), map_location=self.device)
        actor_params = th.load(os.path.join(path, "actor.pth"), map_location=self.device)
        critic_params = th.load(os.path.join(path, "critic.pth"), map_location=self.device)
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(critic_params)
        self.critic_target.load_state_dict(critic_params)
