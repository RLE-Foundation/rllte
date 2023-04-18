from typing import Dict, Tuple
import gymnasium as gym

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributions import Distribution

from hsuanwu.xploit.learner import utils


class StochasticActor(nn.Module):
    """Stochastic actor network for SACLearner. Here the 'self.dist' refers to an sampling distribution instance.

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

    def get_action(self, obs: th.Tensor, step: float = None) -> Distribution:
        """Get actions.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        log_std = th.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )

        std = log_std.exp()

        return self.dist(mu, std)


class DeterministicActor(nn.Module):
    """Deterministic actor network for DrQv2Learner. Here the 'self.dist' refers to an action noise instance.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(
        self, action_space: gym.Space, feature_dim: int = 64, hidden_dim: int = 1024
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
        )
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_action(self, obs: th.Tensor, step: float = None) -> Distribution:
        """Get actions.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Hsuanwu distribution.
        """
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = th.tanh(mu)

        # for Scheduled Exploration Noise
        self.dist.reset(mu, step)

        return self.dist


class DoubleCritic(nn.Module):
    """Double critic network for DrQv2Learner and SACLearner.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(
        self, action_space: gym.Space, feature_dim: int = 64, hidden_dim: int = 1024
    ) -> None:
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

    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor]:
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


class DiscreteActorCritic(nn.Module):
    """Actor-Critic network for discrete control tasks. For PPOLearner, DrACLearner.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic instance.
    """

    def __init__(self, action_space: gym.Space, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.trunk(obs))

    def get_action(self, obs: th.Tensor) -> th.Tensor:
        """Get deterministic actions for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        mu = self.actor(self.trunk(obs))
        return self.dist(mu).mode

    def get_action_and_value(
        self, obs: th.Tensor, actions: th.Tensor = None
    ) -> Tuple[th.Tensor]:
        """Get actions and estimated values for observations.

        Args:
            obs (Tensor): Sampled observations.
            actions (Tensor): Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.trunk(obs)
        mu = self.actor(h)
        dist = self.dist(mu)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy


class DiscreteActorAuxiliaryCritic(nn.Module):
    """Actor-Critic network for discrete control tasks. For PPGLearner.

    Args:
        action_space (Space): Action space of the environment.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic instance.
    """

    def __init__(self, action_space: gym.Space, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)
        self.aux_critic = nn.Linear(hidden_dim, 1)
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.trunk(obs))

    def get_action(self, obs: th.Tensor) -> th.Tensor:
        """Get deterministic actions for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        logits = self.actor(self.trunk(obs))
        return self.dist(logits).mode

    def get_action_and_value(
        self, obs: th.Tensor, actions: th.Tensor = None
    ) -> Tuple[th.Tensor]:
        """Get actions and estimated values for observations.

        Args:
            obs: Sampled observations.
            actions: Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.trunk(obs)
        logits = self.actor(h)
        dist = self.dist(logits)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy

    def get_probs_and_aux_value(self, obs: th.Tensor) -> Tuple[th.Tensor]:
        """Get probs and auxiliary estimated values for auxiliary phase update.

        Args:
            obs: Sampled observations.

        Returns:
            Distribution, estimated values, auxiliary estimated values.
        """
        h = self.trunk(obs)
        logits = self.actor(h)
        dist = self.dist(logits)

        return dist, self.critic(h.detach()), self.aux_critic(h)

    def get_logits(self, obs: th.Tensor) -> Distribution:
        """Get the log-odds of sampling.

        Args:
            obs: Sampled observations.

        Returns:
            Distribution
        """
        return self.dist(self.actor(self.trunk(obs)))


class DiscreteLSTMActor(nn.Module):
    def __init__(
        self,
        action_space,
        feature_dim,
        hidden_dim: int = 512,
        use_lstm: bool = False,
    ) -> None:
        super().__init__()
        """
        Actor network for IMPALA learner that supports LSTM module.

        Args:
            action_space (Space): Action space of the environment.
            feature_dim (int): Number of features accepted.
            hidden_dim (int): Number of units per hidden layer.
            use_lstm (bool): Use LSTM or not.

        Returns:
            Actor network instance.
        """
        self.num_actions = action_space.shape[0]
        self.use_lstm = use_lstm

        # feature_dim + one-hot of last action + last reward
        lstm_output_size = feature_dim + self.num_actions + 1
        if use_lstm:
            self.lstm = nn.LSTM(lstm_output_size, lstm_output_size, 2)

        # policy logits
        self.policy = nn.Linear(lstm_output_size, self.num_actions)
        # baseline value function
        self.baseline = nn.Linear(lstm_output_size, 1)

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
        return tuple(
            th.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
            for _ in range(2)
        )

    def get_action(
        self,
        inputs: Dict,
        lstm_state: Tuple = (),
        training: bool = True,
    ) -> th.Tensor:
        x = inputs["obs"]  # [T, B, *obs_shape], T: rollout length, B: batch size
        T, B, *_ = x.shape
        # TODO: merge time and batch
        x = th.flatten(x, 0, 1)
        # TODO: extract features from observations
        features = F.relu(self.encoder(x))
        # TODO: get one-hot last actions
        one_hot_last_actions = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()

        clipped_reward = th.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        lstm_input = th.cat([features, clipped_reward, one_hot_last_actions], dim=-1)

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

        policy_logits = self.policy(lstm_output)
        baseline = self.baseline(lstm_output)

        if training:
            action = self.dist(policy_logits).sample()
        else:
            action = self.dist(policy_logits).mode

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            lstm_state,
        )
