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


import itertools
from copy import deepcopy
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from rllte.common.on_policy_agent import OnPolicyAgent
from rllte.common.utils import get_network_init


class DAAC(OnPolicyAgent):
    """Decoupled Advantage Actor-Critic (DAAC) agent.
        When 'augmentation' module is invoked, this learner will transform into
        Data Regularized Decoupled Actor-Critic (DrAAC) agent.
        Based on: https://github.com/rraileanu/idaac

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_steps (int): The sample length of per rollout.
        eval_every_episodes (int): Evaluation interval.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.
        hidden_dim (int): The size of the hidden layers.
        clip_range (float): Clipping parameter.
        clip_range_vf (float): Clipping parameter for the value function.
        policy_epochs (int): Times of updating the policy network.
        value_freq (int): Update frequency of the value network.
        value_epochs (int): Times of updating the value network.
        vf_coef (float): Weighting coefficient of value loss.
        ent_coef (float): Weighting coefficient of entropy bonus.
        aug_coef (float): Weighting coefficient of augmentation loss.
        adv_ceof (float): Weighting coefficient of advantage loss.
        max_grad_norm (float): Maximum norm of gradients.
        network_init_method (str): Network initialization method name.

    Returns:
        DAAC agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_steps: int = 128,
        eval_every_episodes: int = 10,
        feature_dim: int = 512,
        batch_size: int = 256,
        lr: float = 2.5e-4,
        eps: float = 1e-5,
        hidden_dim: int = 256,
        clip_range: float = 0.2,
        clip_range_vf: float = 0.2,
        policy_epochs: int = 1,
        value_freq: int = 1,
        value_epochs: int = 9,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        aug_coef: float = 0.1,
        adv_coef: float = 0.25,
        max_grad_norm: float = 0.5,
        network_init_method: str = "xavier_uniform",
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            tag=tag,
            seed=seed,
            device=device,
            pretraining=pretraining,
            num_steps=num_steps,
            eval_every_episodes=eval_every_episodes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            shared_encoder=False,
        )
        self.lr = lr
        self.eps = eps
        self.policy_epochs = policy_epochs
        self.value_freq = value_freq
        self.value_epochs = value_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.aug_coef = aug_coef
        self.adv_coef = adv_coef
        self.max_grad_norm = max_grad_norm
        self.network_init_method = network_init_method
        # training track
        self.num_policy_updates = 0
        self.prev_total_critic_loss = 0

    def freeze(self) -> None:
        """Freeze the structure of the agent."""
        # set encoder and distribution
        self.policy.actor_encoder = self.encoder
        self.policy.critic_encoder = deepcopy(self.encoder)
        self.policy.dist = self.dist
        # network initialization
        self.policy.apply(get_network_init(self.network_init_method))
        # to device
        self.policy.to(self.device)
        # create optimizers
        self.actor_params = itertools.chain(
            self.policy.actor_encoder.parameters(), self.policy.actor.parameters(), self.policy.gae.parameters()
        )
        self.critic_params = itertools.chain(self.policy.critic_encoder.parameters(), self.policy.critic.parameters())
        self.actor_opt = th.optim.Adam(self.actor_params, lr=self.lr, eps=self.eps)
        self.critic_opt = th.optim.Adam(self.critic_params, lr=self.lr, eps=self.eps)
        # set the training mode
        self.mode(training=True)

    def update(self) -> Dict[str, float]:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        total_actor_loss = [0.0]
        total_adv_loss = [0.0]
        total_critic_loss = [0.0]
        total_entropy_loss = [0.0]
        total_aug_loss = [0.0]

        for _ in range(self.policy_epochs):
            generator = self.storage.sample()

            for batch in generator:
                (
                    batch_obs,
                    batch_actions,
                    batch_values,
                    batch_returns,
                    batch_terminateds,
                    batch_truncateds,
                    batch_old_log_probs,
                    adv_targ,
                ) = batch

                # evaluate sampled actions
                new_adv_preds, _, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch_obs, actions=batch_actions)

                # actor loss part
                ratio = th.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * adv_targ
                surr2 = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_targ
                actor_loss = -th.min(surr1, surr2).mean()
                adv_loss = (new_adv_preds.flatten() - adv_targ).pow(2).mean()

                if self.aug is not None:
                    # augmentation loss part
                    batch_obs_aug = self.aug(batch_obs)
                    new_batch_actions, _, _ = self.policy.get_action_and_value(obs=batch_obs)

                    _, _, log_probs_aug, _ = self.policy.evaluate_actions(obs=batch_obs_aug, actions=new_batch_actions)
                    action_loss_aug = -log_probs_aug.mean()
                    aug_loss = self.aug_coef * action_loss_aug
                else:
                    aug_loss = th.scalar_tensor(s=0.0, requires_grad=False, device=adv_loss.device)

                # update
                self.actor_opt.zero_grad(set_to_none=True)
                (adv_loss * self.adv_coef + actor_loss - entropy * self.ent_coef + aug_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_opt.step()

                total_actor_loss.append(actor_loss.item())
                total_adv_loss.append(adv_loss.item())
                total_entropy_loss.append(entropy.item())
                total_aug_loss.append(aug_loss.item())

        if self.num_policy_updates % self.value_freq == 0:
            for _ in range(self.value_epochs):
                generator = self.storage.sample()

                for batch in generator:
                    (
                        batch_obs,
                        batch_actions,
                        batch_values,
                        batch_returns,
                        batch_terminateds,
                        batch_truncateds,
                        batch_old_log_probs,
                        adv_targ,
                    ) = batch

                    # evaluate sampled actions
                    _, new_values, _, _ = self.policy.evaluate_actions(obs=batch_obs, actions=batch_actions)

                    # critic loss part
                    if self.clip_range_vf is None:
                        critic_loss = 0.5 * (new_values.flatten() - batch_returns).pow(2).mean()
                    else:
                        values_clipped = batch_values + (new_values.flatten() - batch_values).clamp(
                            -self.clip_range_vf, self.clip_range_vf
                        )
                        values_losses = (new_values.flatten() - batch_returns).pow(2)
                        values_losses_clipped = (values_clipped - batch_returns).pow(2)
                        critic_loss = 0.5 * th.max(values_losses, values_losses_clipped).mean()

                    if self.aug is not None:
                        # augmentation loss part
                        batch_obs_aug = self.aug(batch_obs)
                        new_batch_actions, new_values, _ = self.policy.get_action_and_value(obs=batch_obs)

                        _, values_aug, _, _ = self.policy.evaluate_actions(obs=batch_obs_aug, actions=new_batch_actions)
                        value_loss_aug = 0.5 * (th.detach(new_values) - values_aug).pow(2).mean()
                        aug_loss = self.aug_coef * value_loss_aug
                    else:
                        aug_loss = th.scalar_tensor(s=0.0, requires_grad=False, device=adv_loss.device)

                    self.critic_opt.zero_grad(set_to_none=True)
                    (critic_loss + aug_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                    self.critic_opt.step()

                    total_critic_loss.append(critic_loss.item())

            self.prev_total_critic_loss = total_critic_loss
        else:
            total_critic_loss = self.prev_total_critic_loss

        self.num_policy_updates += 1

        return {
            "actor_loss": np.mean(total_actor_loss),
            "critic_loss": np.mean(total_critic_loss),
            "entropy": np.mean(total_entropy_loss),
            "aug_loss": np.mean(total_aug_loss),
        }
