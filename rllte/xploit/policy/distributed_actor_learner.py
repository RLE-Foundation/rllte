from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Distribution
from torch.nn import functional as F

from rllte.common.prototype import BasePolicy
from rllte.common.utils import ExportModel
from rllte.xploit.policy.on_policy_shared_actor_critic import BoxActor, DiscreteActor

PolicyOutputs = namedtuple("PolicyOutputs", ["policy_outputs", "baseline", "action"])


class ActorCritic(nn.Module):
    """Actor-Critic network for IMPALA.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_dim (int): Number of neurons for outputting actions.
        action_type (str): Type of actions.
        action_range (Tuple): Range of actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic network.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_dim: int,
        action_type: str,
        action_range: Tuple,
        feature_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        self.action_shape = action_shape
        self.policy_action_dim = action_dim
        self.action_range = action_range
        self.action_type = action_type

        # feature_dim + one-hot of last action + last reward
        mixed_feature_dim = feature_dim + action_dim + 1

        # build actor and critic
        if self.action_type == "Discrete":
            actor_class = DiscreteActor
            self.policy_reshape_dim = action_dim
        elif self.action_type == "Box":
            actor_class = BoxActor
            self.policy_reshape_dim = action_dim * 2
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}.")

        # build actor and critic
        self.actor = actor_class(
            obs_shape=obs_shape, action_dim=action_dim, feature_dim=mixed_feature_dim, hidden_dim=hidden_dim
        )
        # baseline value function
        self.critic = nn.Linear(mixed_feature_dim, 1)

    def forward(self, inputs: Dict[str, th.Tensor], training: bool = True) -> Dict[str, th.Tensor]:
        """Get actions in training.

        Args:
            inputs (Dict[str, th.Tensor]): Inputs data that contains observations, last actions, ...
            training (bool): Whether in training mode.

        Returns:
            Actions.
        """
        # [T, B, *obs_shape], T: rollout length, B: batch size
        x = inputs["observation"]
        T, B, *_ = x.shape
        # merge time and batch
        x = th.flatten(x, 0, 1)
        # extract features from observations
        features = self.encoder(x)
        # get one-hot last actions
        if self.action_type == "Discrete":
            encoded_actions = F.one_hot(inputs["last_action"].view(T * B), self.policy_action_dim).float()
        else:
            encoded_actions = inputs["last_action"].view(T * B, self.policy_action_dim)
        # merge features and one-hot last actions
        mixed_features = th.cat([features, inputs["reward"].view(T * B, 1), encoded_actions], dim=-1)
        # get policy outputs and baseline
        policy_outputs = self.actor.get_policy_outputs(mixed_features)
        baseline = self.critic(mixed_features)
        dist = self.dist(*policy_outputs)

        if training:
            action = dist.sample()
        else:
            action = dist.mean

        # reshape for policy outputs
        policy_outputs = th.cat(policy_outputs, dim=1).view(T, B, self.policy_reshape_dim)
        baseline = baseline.view(T, B)
        if self.action_type == "Discrete":
            action = action.view(T, B, *self.action_shape)
        elif self.action_type == "Box":
            action = action.view(T, B, *self.action_shape).squeeze(0).clamp(*self.action_range)
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}.")

        return dict(policy_outputs=policy_outputs, baseline=baseline, action=action)

    def get_dist(self, outputs: th.Tensor) -> Distribution:
        """Get action distribution.

        Args:
            outputs (th.Tensor): Policy outputs.

        Returns:
            Action distribution.
        """
        if self.action_type == "Discrete":
            return self.dist(outputs)
        elif self.action_type == "Box":
            mu, logstd = outputs.chunk(2, dim=-1)
            return self.dist(mu, logstd.exp())
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}.")


class DistributedActorLearner(BasePolicy):
    """Actor-Learner network for IMPALA.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        init_fn (Optional[str]): Parameters initialization method.
        use_lstm (bool): Whether to use LSTM module.

    Returns:
        Actor-Critic network.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        feature_dim: int,
        hidden_dim: int = 512,
        opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
        opt_kwargs: Optional[Dict[str, Any]] = None,
        init_fn: Optional[str] = None,
        use_lstm: bool = False,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=opt_class,
            opt_kwargs=opt_kwargs,
            init_fn=init_fn,
        )

        # TODO: add support for LSTM
        self.actor = ActorCritic(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dim=self.policy_action_dim,
            action_type=self.action_type,
            action_range=self.action_range,
            feature_dim=self.feature_dim,
        )
        self.learner = ActorCritic(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dim=self.policy_action_dim,
            action_type=self.action_type,
            action_range=self.action_range,
            feature_dim=self.feature_dim,
        )

    def freeze(self, encoder: nn.Module, dist: Distribution) -> None:
        """Freeze all the elements like `encoder` and `dist`.

        Args:
            encoder (nn.Module): Encoder network.
            dist (Distribution): Distribution class.

        Returns:
            None.
        """
        # set encoder
        assert encoder is not None, "Encoder should not be None!"
        self.actor.encoder = encoder
        self.learner.encoder = deepcopy(encoder)
        # set distribution
        assert dist is not None, "Distribution should not be None!"
        self.actor.dist = dist
        self.learner.dist = dist
        # initialize parameters
        self.actor.apply(self.init_fn)
        self.learner.apply(self.init_fn)
        # synchronize the parameters of actor and learner
        self.actor.load_state_dict(self.learner.state_dict())
        # share memory
        self.actor.share_memory()
        # build optimizers
        self.opt = self.opt_class(self.learner.parameters(), **self.opt_kwargs)

    def to(self, device: th.device) -> None:
        """Only move the learner to device, and keep actor in CPU.

        Args:
            device (th.device): Device to use.

        Returns:
            None.
        """
        self.learner.to(device)

    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Save path.

        Returns:
            None.
        """
        export_model = ExportModel(encoder=self.learner.encoder, actor=self.learner.actor)
        th.save(export_model, path / "agent.pth")

    def load(self, path: str, device: th.device) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.
            device (th.device): Device to use.

        Returns:
            None.
        """
        params = th.load(path, map_location=device)
        self.actor.load_state_dict(params)
        self.learner.load_state_dict(params)
