import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple, Union

import gymnasium as gym
import omegaconf
import torch as th
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from hsuanwu.xploit.agent.base import BaseAgent
from hsuanwu.xploit.agent.networks import (DistributedActorCritic, 
                                           get_network_init)

MATCH_KEYS = {
    "trainer": "DistributedTrainer",
    "storage": ["DistributedStorage"],
    "distribution": ["Categorical", "DiagonalGaussian"],
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
        "network_init_method": "identity"
    },
    "storage": {"name": "DistributedStorage", "num_storages": 60, "batch_size": 4},
    ## TODO: xplore part
    "distribution": {"name": "Categorical"},
    "augmentation": {"name": None},
    "reward": {"name": None},
}


class VTraceLoss:
    def __init__(
        self,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
    ) -> None:
        self.dist = None
        self.clip_rho_threshold = clip_rho_threshold
        self.clip_pg_rho_threshold = clip_pg_rho_threshold

    def compute_ISW(self, target_dist, behavior_dist, action):
        log_rhos = target_dist.log_prob(action) - behavior_dist.log_prob(action)
        return th.exp(log_rhos)

    def __call__(self, batch):
        _target_dist = batch["target_dist"]
        _behavior_dist = batch["behavior_dist"]
        _action = batch["action"]
        _baseline = batch["values"]
        _bootstrap_value = batch["bootstrap_value"]
        _values = batch["values"]
        _discounts = batch["discounts"]
        _rewards = batch["reward"]

        with th.no_grad():
            rhos = self.compute_ISW(target_dist=_target_dist, behavior_dist=_behavior_dist, action=_action)
            if self.clip_rho_threshold is not None:
                clipped_rhos = th.clamp(rhos, max=self.clip_rho_threshold)
            else:
                clipped_rhos = rhos
            cs = th.clamp(rhos, max=1.0)
            # Append bootstrapped value to get [v1, ..., v_t+1]
            values_t_plus_1 = th.cat([_values[1:], th.unsqueeze(_bootstrap_value, 0)], dim=0)
            deltas = clipped_rhos * (_rewards + _discounts * values_t_plus_1 - _values)

            acc = th.zeros_like(_bootstrap_value)
            result = []
            for t in range(_discounts.shape[0] - 1, -1, -1):
                acc = deltas[t] + _discounts[t] * cs[t] * acc
                result.append(acc)
            result.reverse()
            vs_minus_v_xs = th.stack(result)

            # Add V(x_s) to get v_s.
            vs = th.add(vs_minus_v_xs, _values)
            # Advantage for policy gradient.
            broadcasted_bootstrap_values = th.ones_like(vs[0]) * _bootstrap_value
            vs_t_plus_1 = th.cat([vs[1:], broadcasted_bootstrap_values.unsqueeze(0)], dim=0)
            if self.clip_pg_rho_threshold is not None:
                clipped_pg_rhos = th.clamp(rhos, max=self.clip_pg_rho_threshold)
            else:
                clipped_pg_rhos = rhos
            pg_advantages = clipped_pg_rhos * (_rewards + _discounts * vs_t_plus_1 - _values)

        pg_loss = -(_target_dist.log_prob(_action) * pg_advantages).sum()
        baseline_loss = F.mse_loss(vs, _baseline, reduction="sum") * 0.5
        entropy_loss = (_target_dist.entropy()).sum()

        return pg_loss, baseline_loss, entropy_loss


class IMPALA(BaseAgent):
    """Importance Weighted Actor-Learner Architecture (IMPALA).

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": action_space.shape, "n": action_space.n, "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        use_lstm (bool): Use LSTM in the policy network or not.
        ent_coef (float): Weighting coefficient of entropy bonus.
        baseline_coef(float): Weighting coefficient of baseline value loss.
        max_grad_norm (float): Maximum norm of gradients.
        discount (float): Discount factor.
        network_init_method (str): Network initialization method name.

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
        network_init_method: str
    ) -> None:
        super().__init__(observation_space, action_space, device, feature_dim, lr, eps)

        self.ent_coef = ent_coef
        self.baseline_coef = baseline_coef
        self.max_grad_norm = max_grad_norm
        self.discount = discount
        self.network_init_method = network_init_method

        self.actor = DistributedActorCritic(
            obs_shape=self.action_type,
            action_shape=self.action_shape,
            action_dim=self.action_dim,
            action_type=self.action_type,
            action_range=self.action_range,
            feature_dim=feature_dim,
            use_lstm=use_lstm,
        )
        self.learner = DistributedActorCritic(
            obs_shape=self.action_type,
            action_shape=self.action_shape,
            action_dim=self.action_dim,
            action_type=self.action_type,
            action_range=self.action_range,
            feature_dim=feature_dim,
            use_lstm=use_lstm,
        )

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

    def integrate(self, **kwargs) -> None:
        """Integrate agent and other modules (encoder, reward, ...) together"""
        # set encoder and distribution
        self.actor.encoder = kwargs["encoder"]
        self.learner.encoder = deepcopy(kwargs["encoder"])
        self.actor.dist = kwargs["dist"]
        self.learner.dist = kwargs["dist"]
        self.dist = kwargs["dist"]
        # network initialization
        self.actor.apply(get_network_init(self.network_init_method))
        self.learner.apply(get_network_init(self.network_init_method))
        # share memory
        self.actor.share_memory()
        # to device
        self.learner.to(self.device)
        # create optimizers
        self.opt = th.optim.RMSprop(
            self.learner.parameters(),
            lr=self.lr,
            eps=self.eps,
        )
        # set lr scheduler
        self.lr_scheduler = th.optim.lr_scheduler.LambdaLR(self.opt, kwargs["lr_lambda"])

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
        lock=threading.Lock(),  # noqa B008
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
        with lock:
            learner_outputs, _ = learner_model.get_action(batch, init_actor_states)

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs["baseline"][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

            discounts = (~batch["terminated"]).float() * cfgs.agent.discount

            batch.update(
                {
                    "discounts": discounts,
                    "bootstrap_value": bootstrap_value,
                    "target_dist": learner_model.get_dist(learner_outputs["policy_outputs"]),
                    "behavior_dist": learner_model.get_dist(batch["policy_outputs"]),
                    "values": learner_outputs["baseline"],
                }
            )

            pg_loss, baseline_loss, entropy_loss = VTraceLoss()(batch)
            total_loss = pg_loss + cfgs.agent.baseline_coef * baseline_loss - cfgs.agent.ent_coef * entropy_loss

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

    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Storage path.

        Returns:
            None.
        """
        th.save(self.learner, path / "agent.pth")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        actor_params = th.load(os.path.join(path, "actor.pth"), map_location=self.device)
        learner_params = th.load(os.path.join(path, "learner.pth"), map_location=self.device)
        self.actor.load_state_dict(actor_params)
        self.learner.load_state_dict(learner_params)
