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


from copy import deepcopy
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Type, Callable, List, Union

import gymnasium as gym
import torch as th
from torch import nn
from torch.nn import functional as F

from rllte.common.base_distribution import BaseDistribution as Distribution
from rllte.common.utils import ExportModel
from rllte.common.base_policy import BasePolicy

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
            obs (th.Tensor): Observations.

        Returns:
            Unnormalized action probabilities.
        """
        logits = self.actor(obs)
        return (logits,)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
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
            obs (th.Tensor): Observations.

        Returns:
            Mean and standard deviation of action distribution.
        """
        mu = self.actor_mu(obs)
        logstd = self.actor_logstd.expand_as(mu)
        return (mu, logstd.exp())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor_mu(obs)


class ActorCritic(nn.Module):
    """Actor network for IMPALA that supports LSTM module.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_dim (int): Number of neurons for outputting actions.
        action_type (str): Type of actions.
        action_range (List): Range of actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        use_lstm (bool): Whether to use LSTM module.

    Returns:
        Actor-Critic network.
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

        self.use_lstm = use_lstm
        # feature_dim + one-hot of last action + last reward
        lstm_output_size = feature_dim + action_dim + 1
        if use_lstm:
            self.lstm = nn.LSTM(lstm_output_size, lstm_output_size, 2)
        
        # build actor and critic
        if self.action_type == "Discrete":
            actor_class = DiscreteActor
            self.policy_reshape_dim = action_dim
        elif self.action_type == "Box":
            actor_class = BoxActor
            self.policy_reshape_dim = action_dim * 2
        else:
            raise NotImplementedError("Unsupported action type!")
        
        # build actor and critic
        self.actor = actor_class(obs_shape=obs_shape, 
                                 action_dim=action_dim, 
                                 feature_dim=lstm_output_size, 
                                 hidden_dim=hidden_dim)

        # baseline value function
        self.critic = nn.Linear(lstm_output_size, 1)

    def init_state(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        """Initialize the state of LSTM.

        Args:
            batch_size (int): Batch size of input data.

        Returns:
            Initial states.
        """
        if not self.use_lstm:
            return tuple()
        return tuple(th.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size) for _ in range(2))

    def forward(
        self,
        inputs: Dict[str, th.Tensor],
        lstm_state: Tuple = (),
        training: bool = True,
    ) -> Tuple[Dict[str, th.Tensor], Tuple[th.Tensor, ...]]:
        """Get actions in training.

        Args:
            inputs (Dict[str, th.Tensor]): Inputs data that contains observations, last actions, ...
            lstm_state (Tuple): Hidden states of LSTM.
            training (bool): Whether in training mode.

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


class DistributedActorLearner(BasePolicy):
    """Actor network for IMPALA that supports LSTM module.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        opt_class (Type[th.optim.Optimizer]): Optimizer class.
        opt_kwargs (Optional[Dict[str, Any]]): Optimizer keyword arguments.
        init_method (Callable): Initialization method.
        use_lstm (bool): Whether to use LSTM module.

    Returns:
        Actor-Critic network.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 feature_dim: int, 
                 hidden_dim: int = 512,
                 opt_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 opt_kwargs: Optional[Dict[str, Any]] = None,
                 init_method: Callable = nn.init.orthogonal_,
                 use_lstm: bool = False,
                 ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=opt_class,
            opt_kwargs=opt_kwargs,
            init_method=init_method)
        
        self.actor = ActorCritic(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dim=self.action_dim,
            action_type=self.action_type,
            action_range=self.action_range,
            feature_dim=self.feature_dim,
            use_lstm=use_lstm,
        )
        self.learner = ActorCritic(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            action_dim=self.action_dim,
            action_type=self.action_type,
            action_range=self.action_range,
            feature_dim=self.feature_dim,
            use_lstm=use_lstm,
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
        self.actor.apply(self.init_method)
        self.learner.apply(self.init_method)
        # share memory
        self.actor.share_memory()
        # build optimizers
        self.opt = self.opt_class(self.learner.parameters(), **self.opt_kwargs)
    
    def act(self,
        inputs: Dict[str, th.Tensor],
        lstm_state: Tuple = (),
        training: bool = True,
    ) -> Tuple[Dict[str, th.Tensor], Tuple[th.Tensor, ...]]:
        """Get actions in training.

        Args:
            inputs (Dict[str, th.Tensor]): Inputs data that contains observations, last actions, ...
            lstm_state (Tuple): Hidden states of LSTM.
            training (bool): Whether in training mode.

        Returns:
            Actions.
        """
        return self.actor(inputs, lstm_state, training=training)
    
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
            raise NotImplementedError("Unsupported action type!")
    
    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Save path.

        Returns:
            None.
        """
        export_model = ExportModel(encoder=self.learner.encoder, actor=self.learner.actor)
        th.save(export_model, path / "agent.pth")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        params = th.load(path, map_location=self.device)
        self.load_state_dict(params)