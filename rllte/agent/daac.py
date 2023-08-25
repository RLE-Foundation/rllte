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
import numpy as np
import torch as th
from torch import nn

from rllte.common.prototype import OnPolicyAgent
from rllte.xploit.encoder import IdentityEncoder, MnihCnnEncoder
from rllte.xploit.policy import OnPolicyDecoupledActorCritic
from rllte.xploit.storage import VanillaRolloutStorage
from rllte.xplore.distribution import (Bernoulli, 
                                       Categorical, 
                                       DiagonalGaussian,
                                       MultiCategorical)


class DAAC(OnPolicyAgent):
    """Decoupled Advantage Actor-Critic (DAAC) agent.
        Based on: https://github.com/rraileanu/idaac

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_steps (int): The sample length of per rollout.
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
        adv_ceof (float): Weighting coefficient of advantage loss.
        max_grad_norm (float): Maximum norm of gradients.
        init_fn (str): Parameters initialization method.

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
        adv_coef: float = 0.25,
        max_grad_norm: float = 0.5,
        init_fn: str = "xavier_uniform",
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            tag=tag,
            seed=seed,
            device=device,
            pretraining=pretraining,
            num_steps=num_steps,
        )

        # hyper parameters
        self.lr = lr
        self.eps = eps
        self.policy_epochs = policy_epochs
        self.value_freq = value_freq
        self.value_epochs = value_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.adv_coef = adv_coef
        self.max_grad_norm = max_grad_norm

        # training track
        self.num_policy_updates = 0
        self.prev_total_critic_loss = 0

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]
            encoder = IdentityEncoder(observation_space=env.observation_space, feature_dim=feature_dim)

        # default distribution
        if self.action_type == "Discrete":
            dist = Categorical
        elif self.action_type == "Box":
            dist = DiagonalGaussian
        elif self.action_type == "MultiBinary":
            dist = Bernoulli
        elif self.action_type == "MultiDiscrete":
            dist = MultiCategorical
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}!")

        # create policy
        policy = OnPolicyDecoupledActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps),
            init_fn=init_fn,
        )

        # default storage
        storage = VanillaRolloutStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            storage_size=self.num_steps,
            num_envs=self.num_envs,
            batch_size=batch_size,
        )

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=dist)

    def update(self) -> Dict[str, float]:
        """Update function that returns training metrics such as policy loss, value loss, etc.."""
        total_policy_loss = [0.0]
        total_adv_loss = [0.0]
        total_value_loss = [0.0]
        total_entropy_loss = [0.0]

        for _ in range(self.policy_epochs):
            for batch in self.storage.sample():
                # evaluate sampled actions
                new_adv_preds, _, new_log_probs, entropy = self.policy.evaluate_actions(
                    obs=batch.observations, actions=batch.actions
                )

                # policy loss part
                ratio = th.exp(new_log_probs - batch.old_log_probs)
                surr1 = ratio * batch.adv_targ
                surr2 = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch.adv_targ
                policy_loss = -th.min(surr1, surr2).mean()
                adv_loss = (new_adv_preds.flatten() - batch.adv_targ).pow(2).mean()

                # update
                self.policy.optimizers['actor_opt'].zero_grad(set_to_none=True)
                (adv_loss * self.adv_coef + policy_loss - entropy * self.ent_coef).backward()
                nn.utils.clip_grad_norm_(self.policy.actor_params, self.max_grad_norm)
                self.policy.optimizers['actor_opt'].step()

                total_policy_loss.append(policy_loss.item())
                total_adv_loss.append(adv_loss.item())
                total_entropy_loss.append(entropy.item())

        if self.num_policy_updates % self.value_freq == 0:
            for _ in range(self.value_epochs):
                for batch in self.storage.sample():
                    # evaluate sampled actions
                    _, new_values, _, _ = self.policy.evaluate_actions(obs=batch.observations, actions=batch.actions)

                    # value loss part
                    if self.clip_range_vf is None:
                        value_loss = 0.5 * (new_values.flatten() - batch.returns).pow(2).mean()
                    else:
                        values_clipped = batch.values + (new_values.flatten() - batch.values).clamp(
                            -self.clip_range_vf, self.clip_range_vf
                        )
                        values_losses = (new_values.flatten() - batch.returns).pow(2)
                        values_losses_clipped = (values_clipped - batch.returns).pow(2)
                        value_loss = 0.5 * th.max(values_losses, values_losses_clipped).mean()

                    # update
                    self.policy.optimizers['critic_opt'].zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.critic_params, self.max_grad_norm)
                    self.policy.optimizers['critic_opt'].step()

                    total_value_loss.append(value_loss.item())

            self.prev_total_critic_loss = total_value_loss
        else:
            total_value_loss = self.prev_total_critic_loss

        self.num_policy_updates += 1

        return {
            "Policy Loss": np.mean(total_policy_loss),
            "Value Loss": np.mean(total_value_loss),
            "Entropy": np.mean(total_entropy_loss),
            "Advantage Loss": np.mean(total_adv_loss),
        }
