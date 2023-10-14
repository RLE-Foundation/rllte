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


from typing import Optional

import gymnasium as gym
import torch as th
from torch.nn import functional as F

from rllte.agent import utils
from rllte.common.prototype import OffPolicyAgent
from rllte.common.type_alias import VecEnv
from rllte.xploit.encoder import IdentityEncoder, TassaCnnEncoder
from rllte.xploit.policy import OffPolicyDoubleActorDoubleCritic
from rllte.xploit.storage import VanillaReplayStorage
from rllte.xplore.distribution import TruncatedNormalNoise


class TD3(OffPolicyAgent):
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

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
        tau: The soft-update rate.
        update_every_steps (int): The agent update frequency.
        discount (float): Discount factor.
        stddev_clip (float): The exploration std clip range.
        init_fn (str): Parameters initialization method.

    Returns:
        DDPG agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
        storage_size: int = 1000000,
        feature_dim: int = 50,
        batch_size: int = 256,
        lr: float = 1e-4,
        eps: float = 1e-8,
        hidden_dim: int = 1024,
        tau: float = 0.01,
        update_every_steps: int = 2,
        discount: float = 0.99,
        stddev_clip: float = 0.3,
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
        self.tau = tau
        self.discount = discount
        self.update_every_steps = update_every_steps
        self.stddev_clip = stddev_clip

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = TassaCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]  # type: ignore
            encoder = IdentityEncoder(
                observation_space=env.observation_space, feature_dim=feature_dim  # type: ignore[assignment]
            )

        # default distribution
        self.action_space: gym.spaces.Box
        dist = TruncatedNormalNoise()

        # create policy
        policy = OffPolicyDoubleActorDoubleCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps),
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

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=dist)

    def update(self) -> None:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        if self.global_step % self.update_every_steps != 0:
            return None

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
        self.update_actor(encoded_obs.detach())

        # udpate actor and critic target
        utils.soft_update_params(self.policy.actor, self.policy.actor_target, self.tau)
        utils.soft_update_params(self.policy.critic, self.policy.critic_target, self.tau)

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
            # sample actions with actor_target
            dist = self.policy.get_dist(next_obs)
            next_actions = dist.sample(clip=self.stddev_clip)
            next_obs_actions = th.concat([next_obs, next_actions], dim=-1)
            target_Q1, target_Q2 = self.policy.critic_target(next_obs_actions)
            target_V = th.min(target_Q1, target_Q2)
            # TODO: add time limit mask
            # target_Q = rewards + (1.0 - terminateds) * (1.0 - truncateds) * self.discount * target_V
            target_Q = rewards + (1.0 - terminateds) * self.discount * target_V

        Q1, Q2 = self.policy.critic(obs, actions)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.policy.optimizers["encoder_opt"].zero_grad(set_to_none=True)
        self.policy.optimizers["critic_opt"].zero_grad(set_to_none=True)
        critic_loss.backward()
        self.policy.optimizers["critic_opt"].step()
        self.policy.optimizers["encoder_opt"].step()

        # record metrics
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/critic_q1", Q1.mean().item())
        self.logger.record("train/critic_q2", Q2.mean().item())
        self.logger.record("train/critic_target_q", target_Q.mean().item())

    def update_actor(self, obs: th.Tensor) -> None:
        """Update the actor network.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            None.
        """
        # sample actions
        dist = self.policy.get_dist(obs)
        actions = dist.sample(clip=self.stddev_clip)
        obs_actions = th.concat([obs, actions], dim=-1)
        Q1, Q2 = self.policy.critic(obs_actions)
        Q = th.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.policy.optimizers["actor_opt"].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy.optimizers["actor_opt"].step()

        # record metrics
        self.logger.record("train/actor_loss", actor_loss.item())
