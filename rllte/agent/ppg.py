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

import numpy as np
import torch as th
from torch import nn

from rllte.common.prototype import OnPolicyAgent
from rllte.common.type_alias import VecEnv
from rllte.xploit.encoder import IdentityEncoder, MnihCnnEncoder
from rllte.xploit.policy import OnPolicySharedActorCritic
from rllte.xploit.storage import VanillaRolloutStorage
from rllte.xplore.distribution import Bernoulli, Categorical, DiagonalGaussian


class PPG(OnPolicyAgent):
    """Phasic Policy Gradient (PPG).
        Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (VecEnv): Vectorized environments for evaluation.
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
        vf_coef (float): Weighting coefficient of value loss.
        ent_coef (float): Weighting coefficient of entropy bonus.
        max_grad_norm (float): Maximum norm of gradients.
        policy_epochs (int): Number of iterations in the policy phase.
        aux_epochs (int): Number of iterations in the auxiliary phase.
        kl_coef (float): Weighting coefficient of divergence loss.
        num_aux_mini_batch (int) Number of mini-batches in auxiliary phase.
        num_aux_grad_accum (int): Number of gradient accumulation for auxiliary phase update.
        discount (float): Discount factor.
        init_fn (str): Parameters initialization method.

    Returns:
        PPG agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
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
        clip_range: float = 0.2,
        clip_range_vf: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        policy_epochs: int = 32,
        aux_epochs: int = 6,
        kl_coef: float = 1.0,
        num_aux_mini_batch: int = 4,
        num_aux_grad_accum: int = 1,
        discount: float = 0.999,
        init_fn: str = "xavier_uniform"
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
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.policy_epochs = policy_epochs
        self.aux_epochs = aux_epochs
        self.kl_coef = kl_coef
        self.num_aux_grad_accum = num_aux_grad_accum
        self.num_aux_mini_batch = num_aux_mini_batch

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]  # type: ignore
            encoder = IdentityEncoder(
                observation_space=env.observation_space, feature_dim=feature_dim  # type: ignore[assignment]
            )

        # default distribution
        if self.action_type == "Discrete":
            dist = Categorical()
        elif self.action_type == "Box":
            dist = DiagonalGaussian()  # type: ignore[assignment]
        elif self.action_type == "MultiBinary":
            dist = Bernoulli()  # type: ignore[assignment]
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
            aux_critic=True,
        )

        # default storage
        storage = VanillaRolloutStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            storage_size=self.num_steps,
            num_envs=self.num_envs,
            batch_size=batch_size,
            discount=discount
        )

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=dist)

        # auxiliary storages
        if self.action_type == "Box":
            self.get_dist_fn = lambda x: self.dist(*x.chunk(2, dim=1))  # type: ignore
            self.policy_outputs_dim = self.policy_action_dim * 2
        else:
            self.get_dist_fn = lambda x: self.dist(x)  # type: ignore
            self.policy_outputs_dim = self.policy_action_dim

        self.num_aux_rollouts = self.num_envs * self.policy_epochs
        self.aux_obs = th.empty(size=(num_steps, self.num_aux_rollouts, *self.obs_shape), device="cpu", dtype=th.float32)
        self.aux_returns = th.empty(size=(num_steps, self.num_aux_rollouts), device="cpu", dtype=th.float32)
        self.aux_policy_outputs = th.empty(
            size=(num_steps, self.num_aux_rollouts, self.policy_outputs_dim), device="cpu", dtype=th.float32
        )

    def update(self) -> None:
        """Update function that returns training metrics such as policy loss, value loss, etc.."""
        # save the observations and returns for auxiliary phase
        idx = int((self.global_episode // self.num_envs) % self.policy_epochs)
        self.aux_obs[:, idx * self.num_envs : (idx + 1) * self.num_envs].copy_(
            self.storage.observations[:-1].clone()  # type: ignore
        )
        self.aux_returns[:, idx * self.num_envs : (idx + 1) * self.num_envs].copy_(self.storage.returns.clone())

        # policy phase
        total_policy_loss = [0.0]
        total_value_loss = [0.0]
        total_entropy_loss = [0.0]

        for batch in self.storage.sample():
            # evaluate sampled actions
            new_values, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch.observations, actions=batch.actions)

            # policy loss part
            ratio = th.exp(new_log_probs - batch.old_log_probs)
            surr1 = ratio * batch.adv_targ
            surr2 = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch.adv_targ
            policy_loss = -th.min(surr1, surr2).mean()

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
            self.policy.optimizers["opt"].zero_grad(set_to_none=True)
            loss = value_loss * self.vf_coef + policy_loss - entropy * self.ent_coef
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizers["opt"].step()

            total_policy_loss.append(policy_loss.item())
            total_value_loss.append(value_loss.item())
            total_entropy_loss.append(entropy.item())

        if (self.global_episode // self.num_envs + 1) % self.policy_epochs != 0:
            # if not auxiliary phase, return train loss directly.
            # record metrics
            self.logger.record("train/policy_loss", np.mean(total_policy_loss))
            self.logger.record("train/value_loss", np.mean(total_value_loss))
            self.logger.record("train/entropy_loss", np.mean(total_entropy_loss))

            return None

        # auxiliary phase
        # recover the old policy
        for i in range(self.policy_epochs):
            with th.no_grad():
                # get policy outputs
                policy_outputs_ = self.policy.get_policy_outputs(
                    self.aux_obs[:, i * self.num_envs : (i + 1) * self.num_envs].to(self.device).view((-1, *self.obs_shape))
                )
                self.aux_policy_outputs[:, i * self.num_envs : (i + 1) * self.num_envs] = policy_outputs_.view(
                    (self.num_steps, self.num_envs, self.policy_outputs_dim)
                )

        total_aux_value_loss = [0.0]
        total_kl_loss = [0.0]

        for _ in range(self.aux_epochs):
            aux_inds = np.arange(self.num_aux_rollouts)
            np.random.shuffle(aux_inds)
            for j in range(0, self.num_aux_rollouts, self.num_aux_mini_batch):
                batch_inds = aux_inds[j : j + self.num_aux_mini_batch]
                batch_aux_obs = self.aux_obs[:, batch_inds].view((-1, *self.obs_shape)).to(self.device)
                batch_aux_returns = self.aux_returns[:, batch_inds].flatten().to(self.device)
                batch_aux_policy_outputs = (
                    self.aux_policy_outputs[:, batch_inds].view((-1, self.policy_outputs_dim)).to(self.device)
                )

                # evaluate the old policy
                new_dist, new_values, new_aux_values = self.policy.get_dist_and_aux_value(batch_aux_obs)
                # get old distributions
                old_dist = self.get_dist_fn(batch_aux_policy_outputs)

                # divergence loss
                kl_loss = th.distributions.kl_divergence(old_dist, new_dist).mean()
                # value loss
                value_loss = 0.5 * ((new_values.flatten() - batch_aux_returns) ** 2).mean()
                aux_value_loss = 0.5 * ((new_aux_values.flatten() - batch_aux_returns) ** 2).mean()
                # total loss
                (value_loss + aux_value_loss + self.kl_coef * kl_loss).backward()

                if (j + 1) % self.num_aux_grad_accum == 0:
                    self.policy.optimizers["opt"].zero_grad(set_to_none=True)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizers["opt"].step()

                total_aux_value_loss.append(value_loss.item())
                total_aux_value_loss.append(aux_value_loss.item())
                total_kl_loss.append(kl_loss.item())

        # record metrics
        self.logger.record("train/aux_value_loss", np.mean(total_aux_value_loss))
        self.logger.record("train/kl_loss", np.mean(total_kl_loss))
