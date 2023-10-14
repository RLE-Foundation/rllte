# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

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


from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from torch.nn import functional as F

from rllte.agent import utils
from rllte.common.prototype import OffPolicyAgent
from rllte.common.type_alias import VecEnv
from rllte.xploit.encoder import IdentityEncoder, MnihCnnEncoder
from rllte.xploit.policy import OffPolicyStochActorDoubleCritic
from rllte.xploit.storage import VanillaReplayStorage
from rllte.xplore.distribution import Categorical


class SACDiscrete(OffPolicyAgent):
    """Soft Actor-Critic Discrete (SAC-Discrete) agent.

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (VecEnv): Vectorized environments for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_init_steps (int): Number of initial exploration steps.
        storage_size (int): The capacity of the storage.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.
        hidden_dim (int): The size of the hidden layers.
        actor_update_freq (int): The actor update frequency (in steps).
        critic_target_tau (float): The critic Q-function soft-update rate.
        critic_target_update_freq (int): The critic Q-function soft-update frequency (in steps).
        betas (Tuple[float]): Coefficients used for computing running averages of gradient and its square.
        temperature (float): Initial temperature coefficient.
        fixed_temperature (bool): Fixed temperature or not.
        target_entropy_ratio (float): Target entropy ratio.
        discount (float): Discount factor.
        init_fn (str): Parameters initialization method.

    Returns:
        PPO agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 10000,
        storage_size: int = 100000,
        feature_dim: int = 50,
        batch_size: int = 256,
        lr: float = 5e-4,
        eps: float = 1e-8,
        hidden_dim: int = 256,
        actor_update_freq: int = 1,
        critic_target_tau: float = 1e-2,
        critic_target_update_freq: int = 4,
        betas: Tuple[float, float] = (0.9, 0.999),
        temperature: float = 0.0,
        fixed_temperature: bool = False,
        target_entropy_ratio: float = 0.98,
        discount: float = 0.99,
        init_fn: str = "orthogonal",
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            tag=tag,
            seed=seed,
            device=device,
            pretraining=pretraining,
            num_init_steps=num_init_steps,
        )

        # hyper parameters
        self.lr = lr
        self.eps = eps
        self.actor_update_freq = actor_update_freq
        self.critic_target_tau = critic_target_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.fixed_temperature = fixed_temperature
        self.discount = discount
        self.betas = betas

        # target entropy
        assert isinstance(self.action_space, gym.spaces.Discrete), "SAC-Discrete only supports discrete action space!"
        self.target_entropy = -np.log(1.0 / self.action_space.n) * target_entropy_ratio
        self.log_alpha = th.tensor(np.log(temperature), device=self.device, requires_grad=True)
        self.log_alpha_opt = th.optim.Adam([self.log_alpha], lr=self.lr, betas=self.betas)

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]  # type: ignore
            encoder = IdentityEncoder(
                observation_space=env.observation_space, feature_dim=feature_dim  # type: ignore[assignment]
            )

        # create policy
        policy = OffPolicyStochActorDoubleCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps, betas=betas),
            init_fn=init_fn,
        )

        # default storage
        storage = VanillaReplayStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            storage_size=storage_size,
            device=device,
            num_envs=self.num_envs,
            batch_size=batch_size,
        )

        # default distribution
        dist = Categorical()

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=dist)

    @property
    def alpha(self) -> th.Tensor:
        """Get the temperature coefficient."""
        return self.log_alpha.exp()

    def update(self) -> None:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        # sample a batch
        batch = self.storage.sample()

        # compute intrinsic rewards
        if self.irs is not None:
            intrinsic_rewards = self.irs.compute_irs(
                samples={
                    "obs": batch.observations.unsqueeze(1),
                    "actions": batch.actions.unsqueeze(1),
                    "next_obs": batch.next_observations.unsqueeze(1),
                },
                step=self.global_step,
            )
            batch = batch._replace(reward=batch.rewards + intrinsic_rewards.to(self.device))

        # encode
        encoded_obs = self.policy.encoder(batch.observations)
        with th.no_grad():
            encoded_next_obs = self.policy.encoder(batch.next_observations)

        # update criitc
        self.update_critic(
            obs=encoded_obs,
            actions=batch.actions,
            rewards=batch.rewards,
            terminateds=batch.terminateds,
            truncateds=batch.truncateds,
            next_obs=encoded_next_obs,
        )

        # update actor (do not udpate encoder)
        if self.global_step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(encoded_obs.detach())

        # udpate critic target
        if self.global_step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.policy.critic, self.policy.critic_target, self.critic_target_tau)

    def deal_with_zero_probs(self, action_probs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Deal with situation of 0.0 probabilities.

        Args:
            action_probs (th.Tensor): Action probabilities.

        Returns:
            Action probabilities and its log values.
        """
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_probs = th.log(action_probs + z)
        return action_probs, log_probs

    def update_critic(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_obs: th.Tensor,
    ) -> None:
        """Update the critic network.

        Args:
            obs (th.Tensor): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Terminateds.
            truncateds (th.Tensor): Truncateds.
            next_obs (th.Tensor): Next observations.

        Returns:
            None.
        """
        with th.no_grad():
            dist = self.policy.get_dist(next_obs)
            # deal with situation of 0.0 probabilities
            action_probs, log_probs = self.deal_with_zero_probs(dist.probs) # type: ignore[attr-defined]
            target_Q1, target_Q2 = self.policy.critic_target(next_obs)
            target_V = (th.min(target_Q1, target_Q2) - self.alpha.detach() * log_probs) * action_probs
            # TODO: add time limit mask
            # target_Q = rewards + (1.0 - terminateds) * (1.0 - truncateds) * self.discount * target_V
            target_Q = rewards + (1.0 - terminateds) * self.discount * target_V.sum(-1, keepdim=True)

        Q1, Q2 = self.policy.critic(obs)
        Q1 = Q1.gather(1, actions.long())
        Q2 = Q2.gather(1, actions.long())
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.policy.optimizers["encoder_opt"].zero_grad()
        self.policy.optimizers["critic_opt"].zero_grad()
        critic_loss.backward()
        self.policy.optimizers["critic_opt"].step()
        self.policy.optimizers["encoder_opt"].step()

        # record metrics
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/critic_q1", Q1.mean().item())
        self.logger.record("train/critic_q2", Q2.mean().item())
        self.logger.record("train/critic_target_q", target_Q.mean().item())

    def update_actor_and_alpha(self, obs: th.Tensor) -> None:
        """Update the actor network and temperature.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            None.
        """
        # sample actions
        dist = self.policy.get_dist(obs)
        action_probs, log_probs = self.deal_with_zero_probs(dist.probs) # type: ignore[attr-defined]
        actor_Q1, actor_Q2 = self.policy.critic(obs)
        actor_Q = th.min(actor_Q1, actor_Q2)

        actor_loss = ((self.alpha.detach() * log_probs - actor_Q) * action_probs).sum(1).mean()

        # optimize actor
        self.policy.optimizers["actor_opt"].zero_grad()
        actor_loss.backward()
        self.policy.optimizers["actor_opt"].step()

        if not self.fixed_temperature:
            # update temperature
            self.log_alpha_opt.zero_grad()
            log_probs_pi = th.sum(log_probs * action_probs, dim=1)
            alpha_loss = (self.alpha * (-log_probs_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_opt.step()
            self.logger.record("train/alpha_loss", alpha_loss.item())

        # record metrics
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/actor_q", actor_Q.mean().item())
        self.logger.record("train/alpha", self.alpha.item())
        self.logger.record("train/target_entropy", self.target_entropy)
        self.logger.record("train/entropy", -log_probs.mean().item())
