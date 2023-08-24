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


import threading
from typing import Dict, Optional, Tuple

import gymnasium as gym
import torch as th
from torch import nn
from torch.nn import functional as F

from rllte.common.prototype import DistributedAgent
from rllte.xploit.encoder import IdentityEncoder, MnihCnnEncoder
from rllte.xploit.policy import DistributedActorLearner
from rllte.xploit.storage import VanillaDistributedStorage
from rllte.xplore.distribution import Categorical, DiagonalGaussian


class VTraceLoss:
    """V-trace loss function.

    Args:
        clip_rho_threshold (float): Clipping coefficient of `rho`.
        clip_pg_rho_threshold (float): Clipping coefficient of policy gradient `rho`.

    Returns:
        V-trace loss instance.
    """

    def __init__(
        self,
        clip_rho_threshold: float = 1.0,
        clip_pg_rho_threshold: float = 1.0,
    ) -> None:
        self.clip_rho_threshold = clip_rho_threshold
        self.clip_pg_rho_threshold = clip_pg_rho_threshold

    def compute_ISW(self, target_dist, behavior_dist, action):
        log_rhos = target_dist.log_prob(action) - behavior_dist.log_prob(action)
        return th.exp(log_rhos)

    def __call__(self, batch):
        _target_dist = batch["target_dist"]
        _behavior_dist = batch["behavior_dist"]
        if batch["actions"].dtype is th.int64:
            _actions = th.flatten(batch["actions"], 1, -1)
        else:
            _actions = batch["actions"]
        _baseline = batch["values"]
        _bootstrap_value = batch["bootstrap_value"]
        _values = batch["values"]
        _discounts = batch["discounts"]
        _rewards = batch["rewards"]

        with th.no_grad():
            rhos = self.compute_ISW(target_dist=_target_dist, behavior_dist=_behavior_dist, action=_actions)
            if self.clip_rho_threshold is not None:
                clipped_rhos = th.clamp(rhos, max=self.clip_rho_threshold)
            else:
                clipped_rhos = rhos
            cs = th.clamp(rhos, max=1.0)
            # append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = th.cat([_values[1:], th.unsqueeze(_bootstrap_value, 0)], dim=0)
            deltas = clipped_rhos * (_rewards + _discounts * values_t_plus_1 - _values)

            acc = th.zeros_like(_bootstrap_value)
            result = []
            for t in range(_discounts.shape[0] - 1, -1, -1):
                acc = deltas[t] + _discounts[t] * cs[t] * acc
                result.append(acc)
            result.reverse()
            vs_minus_v_xs = th.stack(result)

            # add V(x_s) to get v_s
            vs = th.add(vs_minus_v_xs, _values)
            # advantage for policy gradient
            broadcasted_bootstrap_values = th.ones_like(vs[0]) * _bootstrap_value
            vs_t_plus_1 = th.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)
            if self.clip_pg_rho_threshold is not None:
                clipped_pg_rhos = th.clamp(rhos, max=self.clip_pg_rho_threshold)
            else:
                clipped_pg_rhos = rhos
            pg_advantages = clipped_pg_rhos * (_rewards + _discounts * vs_t_plus_1 - _values)

        pg_loss = -(_target_dist.log_prob(_actions) * pg_advantages).sum()
        baseline_loss = F.mse_loss(vs, _baseline, reduction="sum") * 0.5
        entropy_loss = (_target_dist.entropy()).sum()

        return pg_loss, baseline_loss, entropy_loss


class IMPALA(DistributedAgent):
    """Importance Weighted Actor-Learner Architecture (IMPALA).
        Based on: https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/monobeast.py

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.

        num_steps (int): The sample length of per rollout.
        num_actors (int): Number of actors.
        num_learners (int): Number of learners.
        num_storages (int): Number of storages.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.
        hidden_dim (int): The size of the hidden layers.
        use_lstm (bool): Use LSTM in the policy network or not.
        ent_coef (float): Weighting coefficient of entropy bonus.
        baseline_coef(float): Weighting coefficient of baseline value loss.
        max_grad_norm (float): Maximum norm of gradients.
        discount (float): Discount factor.
        init_fn (str): Parameters initialization method.

    Returns:
        IMPALA agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        num_steps: int = 80,
        num_actors: int = 45,
        num_learners: int = 4,
        num_storages: int = 60,
        feature_dim: int = 512,
        batch_size: int = 4,
        lr: float = 4e-4,
        eps: float = 0.01,
        hidden_dim: int = 512,
        use_lstm: bool = False,
        ent_coef: float = 0.01,
        baseline_coef: float = 0.5,
        max_grad_norm: float = 40,
        discount: float = 0.99,
        init_fn: str = "identity",
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            tag=tag,
            seed=seed,
            device=device,
            num_steps=num_steps,
            num_actors=num_actors,
            num_learners=num_learners,
            num_storages=num_storages,
            batch_size=batch_size,
            feature_dim=feature_dim,
            use_lstm=use_lstm
        )
        # hyper parameters
        self.feature_dim = feature_dim
        self.lr = lr
        self.eps = eps
        self.ent_coef = ent_coef
        self.baseline_coef = baseline_coef
        self.max_grad_norm = max_grad_norm
        self.discount = discount

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
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}!")

        # create policy
        policy = DistributedActorLearner(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.RMSprop,
            opt_kwargs=dict(lr=lr, eps=eps),
            init_fn=init_fn,
            use_lstm=use_lstm,
        )

        # default storage
        storage = VanillaDistributedStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            storage_size=self.num_steps,
            num_storages=num_storages,
            batch_size=batch_size,
        )

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, storage=storage, policy=policy, distribution=dist)

    def update(self, batch: Dict, lock=threading.Lock()) -> Dict[str, Tuple]:  # noqa B008
        """Update the learner model.

        Args:
            batch (Batch): Batch samples.
            lock (Lock): Thread lock.

        Returns:
            Training metrics.
        """

        with lock:
            learner_outputs = self.policy.learner(batch)

            # take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baselines"][-1]

            # move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

            discounts = (~batch["terminateds"]).float() * self.discount

            batch.update(
                {
                    "discounts": discounts,
                    "bootstrap_value": bootstrap_value,
                    "target_dist": self.policy.learner.get_dist(learner_outputs["policy_outputs"]),
                    "behavior_dist": self.policy.learner.get_dist(batch["policy_outputs"]),
                    "values": learner_outputs["baselines"],
                }
            )

            pg_loss, baseline_loss, entropy_loss = VTraceLoss()(batch)
            total_loss = pg_loss + self.baseline_coef * baseline_loss - self.ent_coef * entropy_loss

            episode_returns = batch["episode_returns"][batch["terminateds"]]
            episode_steps = batch["episode_steps"][batch["terminateds"]]

            self.policy.optimizers['opt'].zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.learner.parameters(), self.max_grad_norm)
            self.policy.optimizers['opt'].step()
            self.lr_scheduler.step()

            self.policy.actor.load_state_dict(self.policy.learner.state_dict())

            return {
                "episode_returns": tuple(episode_returns.cpu().numpy()),
                "episode_steps": tuple(episode_steps.cpu().numpy()),
                "Total Loss": total_loss.item(),
                "Policy Loss": pg_loss.item(),
                "Value Loss": baseline_loss.item(),
                "Entropy Loss": entropy_loss.item(),
            }
