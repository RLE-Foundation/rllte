import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import gymnasium as gym
import numpy as np
import torch as th
from torch.nn import functional as F

from rllte.common.off_policy_agent import OffPolicyAgent
from rllte.xploit.agent import utils
from rllte.xploit.agent.networks import (ExportModel, 
                                           NpuOffPolicyDeterministicActor, 
                                           OffPolicyDoubleCritic, 
                                           get_network_init)
from rllte.xploit.encoder import TassaCnnEncoder, IdentityEncoder
from rllte.xploit.storage import NStepReplayStorage as Storage
from rllte.xplore.distribution import TruncatedNormalNoise
from rllte.xplore.augmentation import RandomShift


class NpuDrQv2(OffPolicyAgent):
    """Data Regularized-Q v2 (DrQ-v2) for `NPU` device.
        When 'augmentation' module is deprecated, this agent will transform into
            Deep Deterministic Policy Gradient (DDPG) agent.
        Based on: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py

    Args:
        env (Env): A Gym-like environment for training.
        eval_env (Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_init_steps (int): Number of initial exploration steps.
        eval_every_steps (int): Evaluation interval.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.
        hidden_dim (int): The size of the hidden layers.
        critic_target_tau: The critic Q-function soft-update rate.
        update_every_steps (int): The agent update frequency.
        network_init_method (str): Network initialization method name.

    Returns:
        DrQv2 agent instance.
    """

    def __init__(
        self,
        env: gym.Env, 
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
        eval_every_steps: int = 5000,
        feature_dim: int = 50,
        batch_size: int = 256,
        lr: float = 1e-4,
        eps: float = 1e-8,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.01,
        update_every_steps: int = 2,
        network_init_method: str = "orthogonal",
    ) -> None:
        super().__init__(env=env,
                         eval_env=eval_env,
                         tag=tag,
                         seed=seed,
                         device=device,
                         pretraining=pretraining,
                         num_init_steps=num_init_steps,
                         eval_every_steps=eval_every_steps)
        self.feature_dim = feature_dim
        self.lr = lr
        self.eps = eps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.network_init_method = network_init_method

        # build encoder
        if len(self.obs_shape) == 3:
            self.encoder = TassaCnnEncoder(
                observation_space=env.observation_space,
                feature_dim=feature_dim
            )
            self.aug = RandomShift(pad=4)
        elif len(self.obs_shape) == 1:
            self.encoder = IdentityEncoder(
                observation_space=env.observation_space,
                feature_dim=feature_dim
            )
            self.feature_dim = self.obs_shape[0]

        # build storage
        self.storage = Storage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            batch_size=batch_size
        )

        # build distribution
        self.dist = TruncatedNormalNoise()

        # create models
        self.actor = NpuOffPolicyDeterministicActor(
            action_dim=self.action_dim, 
            feature_dim=self.feature_dim, 
            hidden_dim=hidden_dim
        )

        self.critic = OffPolicyDoubleCritic(
            action_dim=self.action_dim, 
            feature_dim=self.feature_dim, 
            hidden_dim=hidden_dim
        )
        
        self.critic_target = OffPolicyDoubleCritic(
            action_dim=self.action_dim, 
            feature_dim=self.feature_dim, 
            hidden_dim=hidden_dim
        )

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        self.critic_target.train(training)

    def set(self, 
            encoder: Optional[Any] = None,
            storage: Optional[Any] = None,
            distribution: Optional[Any] = None,
            augmentation: Optional[Any] = None,
            reward: Optional[Any] = None,
            ) -> None:
        """Set a module for the agent.

        Args:
            encoder (Optional[Any]): An encoder of `rllte.xploit.encoder` or a custom encoder.
            storage (Optional[Any]): A storage of `rllte.xploit.storage` or a custom storage.
            distribution (Optional[Any]): A distribution of `rllte.xplore.distribution` or a custom distribution.
            augmentation (Optional[Any]): An augmentation of `rllte.xplore.augmentation` or a custom augmentation.
            reward (Optional[Any]): A reward of `rllte.xplore.reward` or a custom reward.

        Returns:
            None.
        """
        super().set(
            encoder=encoder,
            storage=storage,
            distribution=distribution,
            augmentation=augmentation,
            reward=reward
        )
        if encoder is not None:
            self.encoder = encoder
            assert self.encoder.feature_dim == self.feature_dim, "The `feature_dim` argument of agent and encoder must be same!"

    def freeze(self) -> None:
        """Freeze the structure of the agent."""
        # set encoder and distribution
        self.encoder = self.encoder
        self.actor.dist = self.dist
        # network initialization
        self.encoder.apply(get_network_init(self.network_init_method))
        self.actor.apply(get_network_init(self.network_init_method))
        self.critic.apply(get_network_init(self.network_init_method))
        # to device
        self.encoder.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        # synchronize critic and target critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        # create optimizers
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=self.lr, eps=self.eps)
        self.actor_opt = th.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.eps)
        self.critic_opt = th.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)
        # set the training mode
        self.mode(training=True)

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

    def update(self) -> Dict[str, float]:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc.
        """
        metrics = {}
        if self.global_step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, discount, next_obs = self.storage.sample(self.global_step)
        obs = obs.float().to(self.device)
        action = action.float().to(self.device)
        reward = reward.float().to(self.device)
        discount = discount.float().to(self.device)
        next_obs = next_obs.float().to(self.device)

        if self.irs is not None:
            intrinsic_reward = self.irs.compute_irs(
                samples={
                    "obs": obs.unsqueeze(1),
                    "actions": action.unsqueeze(1),
                    "next_obs": next_obs.unsqueeze(1),
                },
                step=self.global_step,
            )
            reward += intrinsic_reward.to(self.device)

        # obs augmentation
        if self.aug is not None:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())

        # encode
        encoded_obs = self.encoder(obs)
        with th.no_grad():
            encoded_next_obs = self.encoder(next_obs)

        # update criitc
        metrics.update(self.update_critic(encoded_obs, action, reward, discount, encoded_next_obs))

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor(encoded_obs.detach()))

        # udpate critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def update_critic(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        discount: th.Tensor,
        next_obs: th.Tensor,
    ) -> Dict[str, float]:
        """Update the critic network.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.
            reward (Tensor): Rewards.
            discount (Tensor): discounts.
            next_obs (Tensor): Next observations.

        Returns:
            Critic loss metrics.
        """

        with th.no_grad():
            # sample actions
            dist = self.actor.get_dist(next_obs, step=self.global_step)

            next_action = dist.sample(clip=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = th.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

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
        }

    def update_actor(self, obs: th.Tensor) -> Dict[str, float]:
        """Update the actor network.

        Args:
            obs (Tensor): Observations.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self.actor.get_dist(obs, step=self.global_step)
        action = dist.sample(clip=True)

        Q1, Q2 = self.critic(obs, action)
        Q = th.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return {"actor_loss": actor_loss.item()}

    def save(self) -> None:
        """Save models."""
        if self.pretraining:  # pretraining
            save_dir = Path.cwd() / "pretrained"
            save_dir.mkdir(exist_ok=True)
            th.save(self.encoder.state_dict(), save_dir / "encoder.pth")
            th.save(self.actor.state_dict(), save_dir / "actor.pth")
            th.save(self.critic.state_dict(), save_dir / "critic.pth")
        else:
            save_dir = Path.cwd() / "model"
            save_dir.mkdir(exist_ok=True)
            export_model = ExportModel(encoder=self.encoder, actor=self.actor)
            th.save(export_model, save_dir / "agent.pth")

        self.logger.info(f"Model saved at: {save_dir}")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        self.logger.info(f"Loading Initial Parameters from {path}")
        encoder_params = th.load(os.path.join(path, "encoder.pth"), map_location=self.device)
        actor_params = th.load(os.path.join(path, "actor.pth"), map_location=self.device)
        critic_params = th.load(os.path.join(path, "critic.pth"), map_location=self.device)

        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(critic_params)
        self.critic_target.load_state_dict(critic_params)
