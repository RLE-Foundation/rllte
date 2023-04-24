import collections
import threading
from typing import Dict, Tuple, Union

import gymnasium as gym
import omegaconf
import torch as th
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hsuanwu.xploit.agent.base import BaseAgent
from hsuanwu.xploit.agent.network import DiscreteLSTMActor

MATCH_KEYS = {
    "trainer": "DistributedTrainer",
    "storage": ["DistributedStorage"],
    "distribution": ["Categorical"],
    "augmentation": [],
    "reward": [],
}

DEFAULT_CFGS = {
    ## TODO: Train setup
    "device": "cpu",
    "seed": 1,
    "num_train_steps": 30000000,
    "num_actors": 45,
    "num_learners": 4,
    "num_steps": 80,  # The sample length of per rollout.
    ## TODO: Test setup
    "test_every_steps": 5000,  # only for off-policy algorithms
    "num_test_episodes": 10,
    ## TODO: xploit part
    "encoder": {
        "name": "MnihCnnEncoder",
        "observation_space": dict(),
        "feature_dim": 512,
    },
    "agent": {
        "name": "IMPALA",
        "observation_space": dict(),
        "action_space": dict(),
        "device": str,
        "feature_dim": int,
        "lr": 0.0004,
        "eps": 0.01,
        "use_lstm": False,
        "ent_coef": 0.01,
        "baseline_coef": 0.5,
        "max_grad_norm": 40,
        "discount": 0.99,
    },
    "storage": {"name": "DistributedStorage", "num_storages": 60, "batch_size": 4},
    ## TODO: xplore part
    "distribution": {"name": "Categorical"},
    "augmentation": {"name": None},
    "reward": {"name": None},
}


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


class VTrace(object):
    """Compute V-trace off-policy actor critic targets."""

    def __init__(self) -> None:
        pass

    def action_log_probs(self, policy_logits, actions):
        return -F.nll_loss(
            F.log_softmax(th.flatten(policy_logits, 0, -2), dim=-1),
            th.flatten(actions),
            reduction="none",
        ).view_as(actions)

    def from_logits(
        self,
        behavior_policy_logits,
        target_policy_logits,
        actions,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    ):
        """V-trace for softmax policies."""

        target_action_log_probs = self.action_log_probs(target_policy_logits, actions)
        behavior_action_log_probs = self.action_log_probs(behavior_policy_logits, actions)
        log_rhos = target_action_log_probs - behavior_action_log_probs
        vtrace_returns = self.from_importance_weights(
            log_rhos=log_rhos,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
        )
        return VTraceFromLogitsReturns(
            log_rhos=log_rhos,
            behavior_action_log_probs=behavior_action_log_probs,
            target_action_log_probs=target_action_log_probs,
            **vtrace_returns._asdict(),
        )

    @th.no_grad()
    def from_importance_weights(
        self,
        log_rhos,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    ):
        """V-trace from log importance weights."""
        with th.no_grad():
            rhos = th.exp(log_rhos)
            if clip_rho_threshold is not None:
                clipped_rhos = th.clamp(rhos, max=clip_rho_threshold)
            else:
                clipped_rhos = rhos

            cs = th.clamp(rhos, max=1.0)
            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = th.cat([values[1:], th.unsqueeze(bootstrap_value, 0)], dim=0)
            deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

            acc = th.zeros_like(bootstrap_value)
            result = []
            for t in range(discounts.shape[0] - 1, -1, -1):
                acc = deltas[t] + discounts[t] * cs[t] * acc
                result.append(acc)
            result.reverse()
            vs_minus_v_xs = th.stack(result)

            # Add V(x_s) to get v_s.
            vs = th.add(vs_minus_v_xs, values)

            # Advantage for policy gradient.
            broadcasted_bootstrap_values = th.ones_like(vs[0]) * bootstrap_value
            vs_t_plus_1 = th.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)
            if clip_pg_rho_threshold is not None:
                clipped_pg_rhos = th.clamp(rhos, max=clip_pg_rho_threshold)
            else:
                clipped_pg_rhos = rhos
            pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

            # Make sure no gradients backpropagated through the returned values.
            return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class IMPALA(BaseAgent):
    """Importance Weighted Actor-Learner Architecture (IMPALA).

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        use_lstm (bool): Use LSTM in the policy network or not.
        ent_coef (float): Weighting coefficient of entropy bonus.
        baseline_coef(float): .
        max_grad_norm (float): Maximum norm of gradients.
        discount (float): Discount factor.
    Returns:
        IMPALA distance.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str,
        feature_dim: int,
        lr: float,
        eps: float,
        use_lstm: bool,
        ent_coef: float,
        baseline_coef: float,
        max_grad_norm: float,
        discount: float,
    ) -> None:
        super().__init__(observation_space, action_space, device, feature_dim, lr, eps)

        self.ent_coef = ent_coef
        self.baseline_coef = baseline_coef
        self.max_grad_norm = max_grad_norm
        self.discount = discount

        self.actor = DiscreteLSTMActor(action_space=action_space, feature_dim=feature_dim, use_lstm=use_lstm)

        self.learner = DiscreteLSTMActor(action_space=action_space, feature_dim=feature_dim, use_lstm=use_lstm)

    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.actor.train(training)
        self.learner.train(training)

    def act(self, *kwargs):
        """Sample actions based on observations."""
        return None

    @staticmethod
    def update(
        cfgs: omegaconf.DictConfig,
        actor_model: nn.Module,
        learner_model: nn.Module,
        batch: Dict,
        init_actor_states: Tuple[th.Tensor, ...],
        optimizer: th.optim.Optimizer,
        lr_scheduler: th.optim.lr_scheduler,
        lock=threading.Lock(),
    ) -> Dict[str, Tuple]:
        """
        Update the learner model.

        Args:
            cfgs (DictConfig): Training configs.
            actor_model (NNMoudle): Actor network.
            learner_model (NNMoudle): Learner network.
            batch (Batch): Batch samples.
            init_actor_states (List[Tensor]): Initial states for LSTM.
            optimizer (th.optim.Optimizer): Optimizer.
            lr_scheduler (th.optim.lr_scheduler): Learning rate scheduler.
            lock (Lock): Thread lock.

        Returns:
            Training metrics.
        """

        ###########################################################################
        def compute_policy_gradient_loss(logits, actions, advantages):
            cross_entropy = F.nll_loss(
                F.log_softmax(th.flatten(logits, 0, 1), dim=-1),
                target=th.flatten(actions, 0, 1),
                reduction="none",
            )
            cross_entropy = cross_entropy.view_as(advantages)
            return th.sum(cross_entropy * advantages.detach())

        def compute_baseline_loss(advantages):
            return 0.5 * th.sum(advantages**2)

        def compute_entropy_loss(logits):
            """Return the entropy loss, i.e., the negative entropy of the policy."""
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            return th.sum(policy * log_policy)

        ###########################################################################
        """Performs a learning (optimization) step."""
        with lock:
            learner_outputs, _ = learner_model.get_action(batch, init_actor_states)

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

            rewards = batch["reward"]
            clipped_rewards = th.clamp(rewards, -1, 1)

            discounts = (~batch["terminated"]).float() * cfgs.agent.discount

            vtrace_returns = VTrace().from_logits(
                behavior_policy_logits=batch["policy_logits"],
                target_policy_logits=learner_outputs["policy_logits"],
                actions=batch["action"],
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs["baseline"],
                bootstrap_value=bootstrap_value,
            )

            pg_loss = compute_policy_gradient_loss(
                learner_outputs["policy_logits"],
                batch["action"],
                vtrace_returns.pg_advantages,
            )
            baseline_loss = cfgs.agent.baseline_coef * compute_baseline_loss(vtrace_returns.vs - learner_outputs["baseline"])
            entropy_loss = cfgs.agent.ent_coef * compute_entropy_loss(learner_outputs["policy_logits"])

            total_loss = pg_loss + baseline_loss + entropy_loss

            episode_returns = batch["episode_return"][batch["terminated"]]
            episode_steps = batch["episode_step"][batch["terminated"]]

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(learner_model.parameters(), cfgs.agent.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            actor_model.load_state_dict(learner_model.state_dict())
            return {
                "episode_returns": tuple(episode_returns.cpu().numpy()),
                "episode_steps": tuple(episode_steps.cpu().numpy()),
                "total_loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "baseline_loss": baseline_loss.item(),
                "entropy_loss": entropy_loss.item(),
            }
