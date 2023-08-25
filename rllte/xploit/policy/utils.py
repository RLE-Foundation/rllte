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


from typing import Tuple, Dict, Optional

import torch as th
from torch import nn
from torch.distributions import Distribution
from torch.nn import functional as F

def get_actor(action_type: str, actor_kwargs: Dict) -> nn.Module:
    """Get actor network based on action type.

    Args:
        action_type (str): Type of actions.
        actor_kwargs (Dict): Keyword arguments for actor network.

    Returns:
        Actor class.
    """
    if action_type in ["Discrete", "MultiBinary"]:
        actor_class = OnPolicyDiscreteActor
    elif action_type == "Box":
        actor_class = OnPolicyBoxActor
    elif action_type == "MultiDiscrete":
        actor_class = OnPolicyMultiDiscreteActor
    else:
        raise NotImplementedError(f"Unsupported action type {action_type}!")
    return actor_class(**actor_kwargs)

class OnPolicyDiscreteActor(nn.Module):
    """Actor for `Discrete` and `MultiBinary` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """
    def __init__(self, 
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

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        return (self.actor(obs), )
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor(obs)

class OnPolicyBoxActor(OnPolicyDiscreteActor):
    """Actor for `Box` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """
    def __init__(self, obs_shape: Tuple, action_dim: int, feature_dim: int, hidden_dim: int) -> None:
        super().__init__(obs_shape, action_dim, feature_dim, hidden_dim)
        self.actor_logstd = nn.Parameter(th.ones(1, action_dim))

    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        mu = self.actor(obs)
        logstd = self.actor_logstd.expand_as(mu)
        return (mu, logstd.exp())

class OnPolicyMultiDiscreteActor(OnPolicyDiscreteActor):
    """Actor for `MultiDiscrete` tasks.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.
        nvec (Optional[Tuple[int, ...]]): Number of discrete actions. 
            For `MultiDiscrete` action space only.

    Returns:
        Actor network.
    """
    def __init__(self, 
                 obs_shape: Tuple, 
                 action_dim: int, 
                 feature_dim: int, 
                 hidden_dim: int,
                 nvec: Optional[Tuple[int, ...]] = None
                 ) -> None:
        super().__init__(obs_shape, action_dim, feature_dim, hidden_dim)
        self.nvec = nvec
    
    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        return (th.split(self.actor(obs), self.nvec, dim=1), )

class OnPolicyCritic(nn.Module):
    """Critic for on-policy modules.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network.
    """
    def __init__(self, 
                 obs_shape: Tuple,
                 action_dim: int,
                 feature_dim: int, 
                 hidden_dim: int
                 ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.critic = nn.Linear(feature_dim, 1)
        else:
            # for state-based observations and `IdentityEncoder`
            self.critic = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.critic(obs)

class OnPolicyGAE(nn.Module):
    """Advantage estimator for on-policy modules.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Advantage network.
    """
    def __init__(self, 
                 obs_shape: Tuple,
                 action_dim: int,
                 feature_dim: int, 
                 hidden_dim: int
                 ) -> None:
        super().__init__()
        if len(obs_shape) > 1:
            self.gae = nn.Linear(feature_dim + action_dim, 1)
        else:
            # for state-based observations and `IdentityEncoder`
            self.gae = nn.Sequential(
                nn.Linear(feature_dim + action_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(self, obs_actions: th.Tensor) -> th.Tensor:
        """Get estimated values.

        Args:
            obs_actions (th.Tensor): Observations and actions.

        Returns:
            Estimated values.
        """
        return self.gae(obs_actions)
    
class OffPolicyDoubleCritic(nn.Module):
    """Double critic network for off-policy modules.

    Args:
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(self, action_dim: int, feature_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: th.Tensor, action: th.Tensor) -> Tuple[th.Tensor, ...]:
        """Value estimation.

        Args:
            obs (th.Tensor): Observations.
            action (th.Tensor): Actions.

        Returns:
            Estimated values.
        """
        h_action = th.cat([obs, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    
class DisctributedActorCritic(nn.Module):
    """Actor-Critic network for distributed modules.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_dim (int): Number of neurons for outputting actions.
        action_type (str): Type of actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor-Critic network.
    """

    def __init__(self,
                 obs_shape: Tuple,
                 action_shape: Tuple,
                 action_dim: int,
                 action_type: str,
                 feature_dim: int,
                 hidden_dim: int = 512
                 ) -> None:
        super().__init__()

        self.action_shape = action_shape
        self.policy_action_dim = action_dim
        self.action_type = action_type

        # feature_dim + action_dim + last reward
        # for discrete action space, action_dim is the number of actions
        mixed_feature_dim = feature_dim + action_dim + 1

        assert self.action_type in ["Discrete", "Box"], \
            f"Unsupported action type {self.action_type}!"
        
        if self.action_type == "Box":
            self.policy_reshape_dim = action_dim * 2
        else:
            self.policy_reshape_dim = action_dim

        # build actor and critic
        actor_kwargs = dict(obs_shape=obs_shape,
                            action_dim=action_dim, 
                            feature_dim=mixed_feature_dim, 
                            hidden_dim=hidden_dim)
        # if self.action_type == "MultiDiscrete":
        #     actor_kwargs['nvec'] = self.nvec
        self.actor = get_actor(action_type=self.action_type, actor_kwargs=actor_kwargs)
        # baseline value function
        self.critic = nn.Linear(mixed_feature_dim, 1)

    def forward(self, inputs: Dict[str, th.Tensor], training: bool = True) -> Dict[str, th.Tensor]:
        """Get actions in training.

        Args:
            inputs (Dict[str, th.Tensor]): Input data that contains observations, last actions, ...
            training (bool): Whether in training mode.

        Returns:
            Actions.
        """
        # [T, B, *obs_shape], T: rollout length, B: batch size
        x = inputs["observations"]
        T, B, *_ = x.shape
        # merge time and batch
        x = th.flatten(x, 0, 1)
        # extract features from observations
        features = self.encoder(x)
        # get one-hot last actions
        if self.action_type == "Discrete":
            encoded_actions = F.one_hot(inputs["last_actions"].view(T * B), self.policy_action_dim).float()
        else:
            encoded_actions = inputs["last_actions"].view(T * B, self.policy_action_dim)
        # merge features and one-hot last actions
        mixed_features = th.cat([features, inputs["rewards"].view(T * B, 1), encoded_actions], dim=-1)
        # get policy outputs and baseline
        policy_outputs = self.actor.get_policy_outputs(mixed_features)
        baselines = self.critic(mixed_features)
        dist = self.dist(*policy_outputs)

        if training:
            actions = dist.sample()
        else:
            actions = dist.mean

        # reshape for policy outputs
        policy_outputs = th.cat(policy_outputs, dim=1).view(T, B, self.policy_reshape_dim)
        baselines = baselines.view(T, B)
        if self.action_type == "Discrete":
            actions = actions.view(T, B, *self.action_shape)
        elif self.action_type == "Box":
            actions = actions.view(T, B, *self.action_shape).squeeze(0)#.clamp(*self.action_range)
        else:
            raise NotImplementedError(f"Unsupported action type {self.action_type}!")

        return dict(policy_outputs=policy_outputs, baselines=baselines, actions=actions)

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