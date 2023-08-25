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
from rllte.xploit.policy import OnPolicySharedActorCritic
from rllte.xploit.storage import VanillaRolloutStorage
from rllte.xplore.distribution import (Bernoulli, 
                                       Categorical, 
                                       DiagonalGaussian,
                                       MultiCategorical)

class A2C(OnPolicyAgent):
    """Advantage Actor-Critic (A2C) agent.
        Based on: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (Optional[gym.Env]): A Gym-like environment for evaluation.
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

        n_epochs (int): Times of updating the policy.
        vf_coef (float): Weighting coefficient of value loss.
        ent_coef (float): Weighting coefficient of entropy bonus.
        max_grad_norm (float): Maximum norm of gradients.
        init_fn (str): Parameters initialization method.

    Returns:
        A2C agent instance.
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
        hidden_dim: int = 512,
        n_epochs: int = 4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        init_fn: str = "orthogonal",
    ) -> None:
        super().__init__(
            env=env, eval_env=eval_env, tag=tag, seed=seed, device=device, pretraining=pretraining, num_steps=num_steps
        )

        # hyper parameters
        self.lr = lr
        self.eps = eps
        self.n_epochs = n_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

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
        policy = OnPolicySharedActorCritic(
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
        total_value_loss = [0.0]
        total_entropy_loss = [0.0]

        for _ in range(self.n_epochs):
            for batch in self.storage.sample():
                # evaluate sampled actions
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(
                    obs=batch.observations, actions=batch.actions
                )

                # policy loss part
                policy_loss = -(batch.adv_targ * new_log_probs).mean()

                # value loss part
                value_loss = 0.5 * (new_values.flatten() - batch.returns).pow(2).mean()

                # update
                self.policy.optimizers['opt'].zero_grad(set_to_none=True)
                loss = value_loss * self.vf_coef + policy_loss - entropy * self.ent_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizers['opt'].step()

                total_policy_loss.append(policy_loss.item())
                total_value_loss.append(value_loss.item())
                total_entropy_loss.append(entropy.item())

        return {
            "Policy Loss": np.mean(total_policy_loss),
            "Value Loss": np.mean(total_value_loss),
            "Entropy Loss": np.mean(total_entropy_loss),
        }
