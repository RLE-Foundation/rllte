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


from typing import Tuple

import torch as th
from torch import nn

class OnPolicyActor(nn.Module):
    """Actor for on-policy modules.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_type (str): Type of actions.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Actor network.
    """
    def __init__(self, 
                 obs_shape: Tuple,
                 action_type: str,
                 action_dim: int,
                 feature_dim: int, 
                 hidden_dim: int
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
        
        if action_type == "Box":
            self.actor_logstd = nn.Parameter(th.ones(1, action_dim))
            def _forward(obs):
                mu = self.actor(obs)
                logstd = self.actor_logstd.expand_as(mu)
                return (mu, logstd.exp())
            self._forward = _forward
        else:
            self._forward = lambda obs: (self.actor(obs), )
    
    def get_policy_outputs(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Get policy outputs for training.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Mean and variance of sample distributions.
        """
        return self._forward(obs)
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor(obs)

class OnPolicyCritic(nn.Module):
    """Critic for on-policy modules.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_type (str): Type of actions.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Critic network.
    """
    def __init__(self, 
                 obs_shape: Tuple,
                 action_type: str,
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
        action_type (str): Type of actions.
        action_dim (int): Number of neurons for outputting actions.
        feature_dim (int): Number of features accepted.
        hidden_dim (int): Number of units per hidden layer.

    Returns:
        Advantage network.
    """
    def __init__(self, 
                 obs_shape: Tuple,
                 action_type: str,
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