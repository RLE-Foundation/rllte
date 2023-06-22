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


from typing import Dict, Optional

import gymnasium as gym
import torch as th
from torch.nn import functional as F

from rllte.common.utils import get_network_init
from rllte.common.off_policy_agent import OffPolicyAgent
from rllte.xploit.encoder import IdentityEncoder, TassaCnnEncoder
from rllte.xploit.policy import OffPolicyDeterministicActorDoubleCritic
from rllte.xploit.storage import NStepReplayStorage
from rllte.xplore.distribution import TruncatedNormalNoise
from rllte.xplore.augmentation import RandomShift, Identity
from rllte.agent import utils

class DrQv2(OffPolicyAgent):
    """Proximal Policy Optimization (PPO) agent.
        When the `augmentation` module is invoked, this agent will transform into Data Regularized Actor-Critic (DrAC) agent.
        Based on: https://github.com/yuanmingqi/pytorch-a2c-ppo-acktr-gail

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
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
        super().__init__(
            env=env,
            eval_env=eval_env,
            tag=tag,
            seed=seed,
            device=device,
            pretraining=pretraining,
            num_init_steps=num_init_steps,
            eval_every_steps=eval_every_steps
        )

        # hyper parameters
        self.lr = lr
        self.eps = eps
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.network_init_method = network_init_method

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = TassaCnnEncoder(observation_space=env.observation_space, 
                                           feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]
            encoder = IdentityEncoder(observation_space=env.observation_space, 
                                           feature_dim=feature_dim)

        # default distribution
        dist = TruncatedNormalNoise(low=self.action_range[0], high=self.action_range[1])

        # create policy
        policy = OffPolicyDeterministicActorDoubleCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps),
            init_method=get_network_init(self.network_init_method),
        )

        # default storage
        storage = NStepReplayStorage(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device,
                batch_size=batch_size,
        )

        # default augmentation
        # for `DDPG` without augmentation
        if len(self.obs_shape) == 1:
            aug = Identity().to(self.device)
        else:
            aug = RandomShift(pad=4).to(self.device)

        # set all the modules [essential operation!!!]
        self.set(
            encoder=encoder,
            policy=policy,
            storage=storage,
            distribution=dist,
            augmentation=aug
        )


    def update(self) -> Dict[str, float]:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        metrics = {}
        if self.global_step % self.update_every_steps != 0:
            return metrics

        batch = self.storage.sample(self.global_step)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

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
        encoded_obs = self.policy.encoder(obs)
        with th.no_grad():
            encoded_next_obs = self.policy.encoder(next_obs)

        # update criitc
        metrics.update(self.update_critic(encoded_obs, action, reward, discount, encoded_next_obs))

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor(encoded_obs.detach()))

        # udpate critic target
        utils.soft_update_params(self.policy.critic, self.policy.critic_target, self.critic_target_tau)

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
            obs (th.Tensor): Observations.
            action (th.Tensor): Actions.
            reward (th.Tensor): Rewards.
            discount (th.Tensor): discounts.
            next_obs (th.Tensor): Next observations.

        Returns:
            Critic loss metrics.
        """

        with th.no_grad():
            # sample actions
            dist = self.policy.get_dist(next_obs, step=self.global_step)

            next_action = dist.sample(clip=True)
            target_Q1, target_Q2 = self.policy.critic_target(next_obs, next_action)
            target_V = th.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.policy.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.policy.encoder_opt.zero_grad(set_to_none=True)
        self.policy.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.policy.critic_opt.step()
        self.policy.encoder_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "critic_q1": Q1.mean().item(),
            "critic_q2": Q2.mean().item(),
            "critic_target": target_Q.mean().item(),
        }

    def update_actor(self, obs: th.Tensor) -> Dict[str, float]:
        """Update the actor network.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self.policy.get_dist(obs, step=self.global_step)
        action = dist.sample(clip=True)

        Q1, Q2 = self.policy.critic(obs, action)
        Q = th.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.policy.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy.actor_opt.step()

        return {"actor_loss": actor_loss.item()}
