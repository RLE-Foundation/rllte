from typing import Dict, List, Tuple
from pathlib import Path
import os
import torch as th
from torch import nn
from torch.distributions import Distribution
from torch.nn import functional as F
from rllte.common.utils import ExportModel


class DiscreteActor(nn.Module):
    """Actor for `Discrete` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.actor = nn.Linear(feature_dim, action_dim)
        else:
            # for state-based observations and `IdentityEncoder`
            self.actor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (Tensor): Observations.

        Returns:
            Unnormalized probabilities.
        """
        logits = self.actor(obs)
        return (logits,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (Tensor): Observations.

        Returns:
            Unnormalized action probabilities.
        """
        return self.actor(obs)


class BoxActor(nn.Module):
    """Actor for `Box` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.actor_mu = nn.Linear(feature_dim, action_dim)
        else:
            # for state-based observations and `IdentityEncoder`
            self.actor_mu = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
        self.actor_logstd = nn.Parameter(th.ones(1, action_dim))

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        mu = self.actor_mu(obs)
        logstd = self.actor_logstd.expand_as(mu)
        return (mu, logstd.exp())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor_mu(obs)


class DistributedActorCritic(nn.Module):
    """Actor network for IMPALA that supports LSTM module.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_dim (int): Number of neurons for outputting actions.
        action_type (str): The action type like 'Discrete' or 'Box', etc.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        use_lstm (bool): Use LSTM or not.

    Returns:
        Actor network instance.
    """

    def __init__(
        self,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_dim: int,
        action_type: str,
        action_range: List,
        feature_dim: int,
        hidden_dim: int = 512,
        use_lstm: bool = False,
    ) -> None:
        super().__init__()

        self.use_lstm = use_lstm
        self.action_shape = action_shape
        self.action_dim = action_dim
        self.action_range = action_range
        self.action_type = action_type

        # feature_dim + one-hot of last action + last reward
        lstm_output_size = feature_dim + self.action_dim + 1
        if use_lstm:
            self.lstm = nn.LSTM(lstm_output_size, lstm_output_size, 2)

        if action_type == "Discrete":
            self.actor = DiscreteActor(
                obs_shape=obs_shape, action_dim=action_dim, feature_dim=lstm_output_size, hidden_dim=hidden_dim
            )
            self.policy_reshape_dim = self.action_dim

        elif action_type == "Box":
            self.actor = BoxActor(
                obs_shape=obs_shape, action_dim=action_dim, feature_dim=lstm_output_size, hidden_dim=hidden_dim
            )
            self.policy_reshape_dim = self.action_dim * 2
        else:
            raise NotImplementedError("Unsupported action type!")

        # baseline value function
        self.critic = nn.Linear(lstm_output_size, 1)
        # internal encoder
        self.encoder = None
        self.dist = None

    def init_state(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        """Generate the initial states for LSTM.

        Args:
            batch_size (int): The batch size for training.

        Returns:
            Initial states.
        """
        if not self.use_lstm:
            return tuple()
        return tuple(th.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))

    def get_action(
        self,
        inputs: Dict,
        lstm_state: Tuple = (),
        training: bool = True,
    ) -> th.Tensor:
        """Get actions in training.

        Args:
            inputs (Dict): Inputs data that contains observations, last actions, ...
            lstm_state (Tuple): LSTM states.
            training (bool): Training flag.

        Returns:
            Actions.
        """
        x = inputs["obs"]  # [T, B, *obs_shape], T: rollout length, B: batch size
        T, B, *_ = x.shape
        # TODO: merge time and batch
        x = th.flatten(x, 0, 1)
        # TODO: extract features from observations
        features = self.encoder(x)
        # TODO: get one-hot last actions

        if self.action_type == "Discrete":
            encoded_actions = F.one_hot(inputs["last_action"].view(T * B), self.action_dim).float()
        else:
            encoded_actions = inputs["last_action"].view(T * B, self.action_dim)

        lstm_input = th.cat([features, inputs["reward"].view(T * B, 1), encoded_actions], dim=-1)

        if self.use_lstm:
            lstm_input = lstm_input.view(T, B, -1)
            lstm_output_list = []
            notdone = (~inputs["terminated"]).float()
            for input, nd in zip(lstm_input.unbind(), notdone.unbind()):
                # Reset lstm state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                lstm_state = tuple(nd * s for s in lstm_state)
                output, lstm_state = self.lstm(input.unsqueeze(0), lstm_state)
                lstm_output_list.append(output)
            lstm_output = th.flatten(th.cat(lstm_output_list), 0, 1)
        else:
            lstm_output = lstm_input
            lstm_state = tuple()

        policy_outputs = self.actor.get_policy_outputs(lstm_output)
        baseline = self.critic(lstm_output)
        dist = self.dist(*policy_outputs)

        if training:
            action = dist.sample()
        else:
            action = dist.mean

        policy_outputs = th.cat(policy_outputs, dim=1).view(T, B, self.policy_reshape_dim)
        baseline = baseline.view(T, B)

        if self.action_type == "Discrete":
            action = action.view(T, B, *self.action_shape)
        elif self.action_type == "Box":
            action = action.view(T, B, *self.action_shape).squeeze(0).clamp(*self.action_range)
        else:
            raise NotImplementedError("Unsupported action type!")

        return (dict(policy_outputs=policy_outputs, baseline=baseline, action=action), lstm_state)

    def get_dist(self, outputs: th.Tensor) -> Distribution:
        """Get sample distributions.

        Args:
            outputs (Tensor): Policy outputs.

        Returns:
            Sample distributions.
        """
        if self.action_type == "Discrete":
            return self.dist(outputs)
        elif self.action_type == "Box":
            mu, logstd = outputs.chunk(2, dim=-1)
            return self.dist(mu, logstd.exp())
        else:
            raise NotImplementedError("Unsupported action type!")

    def forward(
        self,
        inputs: Dict,
        lstm_state: Tuple = (),
    ) -> th.Tensor:
        """Get actions in training.

        Args:
            inputs (Dict): Inputs data that contains observations, last actions, ...
            lstm_state (Tuple): LSTM states.
            training (bool): Training flag.

        Returns:
            Actions.
        """
    
    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Save path.

        Returns:
            None.
        """
        export_model = ExportModel(encoder=self.encoder, actor=self.actor)
        th.save(export_model, path / "agent.pth")
    
    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        params = th.load(os.path.join(path, "pretrained.pth"), map_location=self.device)
        self.load_state_dict(params)
