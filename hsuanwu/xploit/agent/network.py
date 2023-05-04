from typing import Dict, Tuple, List

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Distribution
from torch.nn import functional as F

from hsuanwu.xploit.agent import utils


class OffPolicyStochasticActor(nn.Module):
    """Stochastic actor network for SAC. Here the 'self.dist' refers to an sampling distribution instance.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(
        self,
        action_space: gym.Space,
        feature_dim: int = 64,
        hidden_dim: int = 1024,
        log_std_range: Tuple = (-10, 2),
    ) -> None:
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_space.shape[0]),
        )
        # placeholder for distribution
        self.dist = None
        self.log_std_min, self.log_std_max = log_std_range

        self.apply(utils.network_init)

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        log_std = th.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()

        return self.dist(mu, std)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Get deterministic actions.

        Args:
            obs (Tensor): Observations.

        Returns:
            Actions.
        """
        mu, _ = self.policy(obs).chunk(2, dim=-1)
        return mu


class OffPolicyDeterministicActor(nn.Module):
    """Deterministic actor network for DrQv2. Here the 'self.dist' refers to an action noise instance.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(self, action_space: gym.Space, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
            nn.Tanh(),
        )
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_dist(self, obs: th.Tensor, step: int) -> Distribution:
        """Get sample distribution.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        mu = self.policy(obs)

        # for Scheduled Exploration Noise
        self.dist.reset(mu, step)

        return self.dist

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Get actions.

        Args:
            obs (Tensor): Observations.

        Returns:
            Actions.
        """
        return self.policy(obs)


class OffPolicyDoubleCritic(nn.Module):
    """Double critic network for DrQv2 and SAC.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(self, action_space: gym.Space, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()

        action_shape = action_space.shape
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.network_init)

    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Value estimation.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.

        Returns:
            Estimated values.
        """
        h_action = th.cat([obs, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class OnPolicySharedActorCritic(nn.Module):
    """Actor-Critic network using a shared encoder for on-policy algorithms.

    Args:
        action_shape (Tuple): The data shape of actions.
        action_type (str): The action type like 'Discrete' or 'Box', etc.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        aux_critic (bool): Use auxiliary critic or not.

    Returns:
        Actor-Critic instance.
    """

    class DiscreteActor(nn.Module):
        """Actor for 'Discrete' tasks."""

        def __init__(self, action_shape, hidden_dim) -> None:
            super().__init__()
            self.actor = nn.Linear(hidden_dim, action_shape[0])

        def get_policy_outputs(self, obs: th.Tensor) -> th.Tensor:
            logits = self.actor(obs)
            return (logits,)

        def forward(self, obs: th.Tensor) -> th.Tensor:
            """Only for model inference"""
            return self.actor(obs)

    class BoxActor(nn.Module):
        """Actor for 'Box' tasks."""

        def __init__(self, action_shape, hidden_dim) -> None:
            super().__init__()
            self.actor_mu = nn.Linear(hidden_dim, action_shape[0])
            self.actor_logstd = nn.Parameter(th.zeros(1, action_shape[0]))

        def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
            mu = self.actor_mu(obs)
            logstd = self.actor_logstd.expand_as(mu)
            return (mu, logstd.exp())

        def forward(self, obs: th.Tensor) -> th.Tensor:
            """Only for model inference"""
            return self.actor_mu(obs)
    
    class MultiBinaryActor(nn.Module):
        """Actor for 'MultiBinary' tasks."""

        def __init__(self, action_shape, hidden_dim) -> None:
            super().__init__()
            self.actor = nn.Linear(hidden_dim, action_shape[0])

        def get_policy_outputs(self, obs: th.Tensor) -> th.Tensor:
            logits = self.actor(obs)
            return (logits,)

        def forward(self, obs: th.Tensor) -> th.Tensor:
            """Only for model inference"""
            return self.actor(obs)

    def __init__(
        self,
        action_shape: Tuple,
        action_type: str,
        feature_dim: int,
        hidden_dim: int,
        aux_critic: bool = False,
    ) -> None:
        super().__init__()

        # self.trunk = nn.Sequential(
        #     nn.LayerNorm(feature_dim),
        #     nn.Tanh(),
        #     nn.Linear(feature_dim, hidden_dim),
        #     nn.ReLU(),
        # )
        if action_type == "Discrete":
            self.actor = self.DiscreteActor(action_shape=action_shape, hidden_dim=feature_dim)
        elif action_type == "Box":
            self.actor = self.BoxActor(action_shape=action_shape, hidden_dim=feature_dim)
        elif action_type == "MultiBinary":
            self.actor = self.MultiBinaryActor(action_shape=action_shape, hidden_dim=feature_dim)
        else:
            raise NotImplementedError("Unsupported action type!")

        self.critic = nn.Linear(hidden_dim, 1)
        if aux_critic:
            self.aux_critic = nn.Linear(hidden_dim, 1)

        # placeholder for distribution
        self.encoder = None
        self.dist = None

        self.apply(utils.network_init)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference

        Args:
            obs (Tensor): Observations.

        Returns:
            Actions.
        """
        return self.actor(self.encoder(obs))

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.encoder(obs))

    def get_det_action(self, obs: th.Tensor) -> th.Tensor:
        """Get deterministic actions for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        policy_outputs = self.actor.get_policy_outputs(self.encoder(obs))
        dist = self.dist(*policy_outputs)

        return dist.mean

    def get_action_and_value(self, obs: th.Tensor, actions: th.Tensor = None) -> Tuple[th.Tensor, ...]:
        """Get actions and estimated values for observations.

        Args:
            obs (Tensor): Sampled observations.
            actions (Tensor): Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy

    def get_probs_and_aux_value(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Get probs and auxiliary estimated values for auxiliary phase update.

        Args:
            obs: Sampled observations.

        Returns:
            Distribution, estimated values, auxiliary estimated values.
        """
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        dist = self.dist(*policy_outputs)

        return dist, self.critic(h.detach()), self.aux_critic(h)

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor]:
        h = self.encoder(obs)
        policy_outputs = self.actor.get_policy_outputs(h)
        return th.cat(policy_outputs, dim=1)


class DistributedActorCritic(nn.Module):
    """Actor network for IMPALA that supports LSTM module.

    Args:
        action_shape (Tuple): The data shape of actions.
        action_type (str): The action type like 'Discrete' or 'Box', etc.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        use_lstm (bool): Use LSTM or not.

    Returns:
        Actor network instance.
    """ 
    class DiscreteActor(nn.Module):
        """Actor for 'Discrete' tasks."""

        def __init__(self, action_shape, hidden_dim) -> None:
            super().__init__()
            self.actor = nn.Linear(hidden_dim, action_shape[0])

        def get_policy_outputs(self, obs: th.Tensor) -> th.Tensor:
            logits = self.actor(obs)
            return (logits,)

        def forward(self, obs: th.Tensor) -> th.Tensor:
            """Only for model inference"""
            return self.actor(obs)

    class BoxActor(nn.Module):
        """Actor for 'Box' tasks."""

        def __init__(self, action_shape, hidden_dim) -> None:
            super().__init__()
            self.actor_mu = nn.Linear(hidden_dim, action_shape[0])
            self.actor_logstd = nn.Parameter(th.zeros(1, action_shape[0]))

        def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
            mu = self.actor_mu(obs)
            logstd = self.actor_logstd.expand_as(mu)
            return (mu, logstd.exp())

        def forward(self, obs: th.Tensor) -> th.Tensor:
            """Only for model inference"""
            return self.actor_mu(obs)
        
    def __init__(
        self,
        action_shape: Tuple,
        action_type: str,
        action_range: List,
        feature_dim,
        hidden_dim: int = 512,
        use_lstm: bool = False,
    ) -> None:
        super().__init__()

        self.num_actions = action_shape[0]
        self.use_lstm = use_lstm
        self.action_range = action_range
        self.action_type = action_type

        # feature_dim + one-hot of last action + last reward
        lstm_output_size = feature_dim + self.num_actions + 1
        if use_lstm:
            self.lstm = nn.LSTM(lstm_output_size, lstm_output_size, 2)
        
        if action_type == "Discrete":
            self.actor = self.DiscreteActor(action_shape=action_shape, 
                                            hidden_dim=lstm_output_size)
            self.action_dim = 1
            self.policy_reshape_dim = action_shape[0]
        elif action_type == "Box":
            self.actor = self.BoxActor(action_shape=action_shape, 
                                       hidden_dim=lstm_output_size)
            self.action_dim = action_shape[0]
            self.policy_reshape_dim = action_shape[0] * 2
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
    
    def get_action(self,
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
            encoded_actions = F.one_hot(inputs["last_action"].view(T * B), self.num_actions).float()
        else:
            encoded_actions = inputs["last_action"].view(T * B, self.num_actions)

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
            action = action.view(T, B)
        elif self.action_type == "Box":
            action = action.view(T, B, self.num_actions).squeeze(0).clamp(*self.action_range)
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
    
    def forward(self,
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

        