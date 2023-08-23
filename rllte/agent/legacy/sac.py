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


from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from torch.nn import functional as F

from rllte.agent import utils
from rllte.common.prototype import OffPolicyAgent
from rllte.xploit.encoder import IdentityEncoder, TassaCnnEncoder
from rllte.xploit.policy import OffPolicyStochActorDoubleCritic
from rllte.xploit.storage import VanillaReplayStorage
from rllte.xplore.distribution import SquashedNormal


class SAC(OffPolicyAgent):
    """Soft Actor-Critic (SAC) agent.
        Based on: https://github.com/denisyarats/pytorch_sac

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (Optional[gym.Env]): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_init_steps (int): Number of initial exploration steps.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
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
        init_fn (str): Parameters initialization method.

    Returns:
        PPO agent instance.
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
        feature_dim: int = 50,
        batch_size: int = 1024,
        lr: float = 1e-4,
        eps: float = 1e-8,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.005,
        update_every_steps: int = 2,
        log_std_range: Tuple[float, ...] = (-5.0, 2),
        betas: Tuple[float, ...] = (0.9, 0.999),
        temperature: float = 0.1,
        fixed_temperature: bool = False,
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
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.fixed_temperature = fixed_temperature
        self.discount = discount
        self.betas = betas

        # target entropy
        self.target_entropy = -self.action_dim
        self.log_alpha = th.tensor(np.log(temperature), device=self.device, requires_grad=True)
        self.log_alpha_opt = th.optim.Adam([self.log_alpha], lr=self.lr, betas=self.betas)

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = TassaCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]
            encoder = IdentityEncoder(observation_space=env.observation_space, feature_dim=feature_dim)

        # default distribution
        dist = SquashedNormal

        # create policy
        policy = OffPolicyStochActorDoubleCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps, betas=betas),
            log_std_range=log_std_range,
            init_fn=init_fn,
        )

        # default storage
        storage = VanillaReplayStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_envs=self.num_envs,
            batch_size=batch_size,
        )

        # set all the modules [essential operation!!!]
        self.set(
            encoder=encoder,
            policy=policy,
            storage=storage,
            distribution=dist,
        )

    @property
    def alpha(self) -> th.Tensor:
        """Get the temperature coefficient."""
        return self.log_alpha.exp()

    def update(self) -> Dict[str, float]:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        metrics = {}
        if self.global_step % self.update_every_steps != 0:
            return metrics

        # sample a batch
        batch = self.storage.sample()

        # compute intrinsic rewards
        if self.irs is not None:
            intrinsic_rewards = self.irs.compute_irs(
                samples={
                    "obs": batch.observations,
                    "actions": batch.actions,
                    "next_obs": batch.next_observations,
                },
                step=self.global_step,
            )
            batch = batch._replace(reward=batch.rewards + intrinsic_rewards.to(self.device))

        # encode
        encoded_obs = self.policy.encoder(batch.observations)
        with th.no_grad():
            encoded_next_obs = self.policy.encoder(batch.next_observations)

        # update criitc
        metrics.update(
            self.update_critic(
                obs=encoded_obs,
                actions=batch.actions,
                rewards=batch.rewards,
                terminateds=batch.terminateds,
                truncateds=batch.truncateds,
                next_obs=encoded_next_obs,
            )
        )

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor_and_alpha(encoded_obs.detach()))

        # udpate critic target
        utils.soft_update_params(self.policy.critic, self.policy.critic_target, self.critic_target_tau)

        return metrics

    def update_critic(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_obs: th.Tensor,
    ) -> Dict[str, float]:
        """Update the critic network.

        Args:
            obs (th.Tensor): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Terminateds.
            truncateds (th.Tensor): Truncateds.
            next_obs (th.Tensor): Next observations.

        Returns:
            Critic loss.
        """
        with th.no_grad():
            dist = self.policy.get_dist(next_obs, step=self.global_step)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.policy.critic_target(next_obs, next_action)
            target_V = th.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            # time limit mask
            target_Q = rewards + (1.0 - terminateds) * (1.0 - truncateds) * self.discount * target_V

        Q1, Q2 = self.policy.critic(obs, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.policy.optimizers['encoder_opt'].zero_grad(set_to_none=True)
        self.policy.optimizers['critic_opt'].zero_grad(set_to_none=True)
        critic_loss.backward()
        self.policy.optimizers['critic_opt'].step()
        self.policy.optimizers['encoder_opt'].step()

        return {
            "Critic Loss": critic_loss.item(),
            "Q1": Q1.mean().item(),
            "Q2": Q2.mean().item(),
            "Target Q": target_Q.mean().item(),
        }

    def update_actor_and_alpha(self, obs: th.Tensor) -> Dict[str, float]:
        """Update the actor network and temperature.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Policy loss.
        """
        # sample actions
        dist = self.policy.get_dist(obs, step=self.global_step)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.policy.critic(obs, action)
        Q = th.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize actor
        self.policy.optimizers['actor_opt'].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy.optimizers['actor_opt'].step()

        if not self.fixed_temperature:
            # update temperature
            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_opt.step()
        else:
            alpha_loss = th.scalar_tensor(s=0.0)

        return {"Actor Loss": actor_loss.item(), "Alpha Loss": alpha_loss.item()}
