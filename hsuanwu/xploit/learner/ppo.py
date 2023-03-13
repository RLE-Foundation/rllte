from torch.nn import functional as F
from torch import nn
import torch

from hsuanwu.common.typing import *
from hsuanwu.xploit import utils


class ActorCritic(nn.Module):
    """Actor-Critic module.
    
    Args:
        action_space: Action space of the environment.
        feature_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.

    Returns:
        Actor-Critic instance.
    """
    def __init__(self,
                 action_space: Space,
                 feature_dim: int,
                 hidden_dim: int
                 ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Linear(hidden_dim, action_space.n)
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(utils.network_init)
    
    def forward(self, obs: Tensor) -> Sequence[Tensor]:
        feature = self.trunk(obs)
        mu = self.actor(feature)
        value = self.critic(feature)

        return mu, value


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent
        Paper: Proximal policy optimization algorithms
        Link: https://arxiv.org/pdf/1707.06347
    
    Args:
        observation_space: Observation space of the environment.
        action_space: Action shape of the environment.
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        hidden_dim: The size of the hidden layers.
        lr: The learning rate.
    
    Returns:
        Agent instance.
    """
    def __init__(self,
                 observation_space: Space, 
                 action_space: Space,
                 action_type: str,
                 device: torch.device = 'cuda',
                 feature_dim: int = 50,
                 hidden_dim: int = 1024,
                 lr: float = 1e-4,) -> None:
        self._action_type = action_type
        self._device = torch.device(device)
        self._lr = lr

        # create models
        self._encoder = None
        self._actor_critc = ActorCritic(
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        ).to(self._device)

        # create optimizers
        self._actor_critc_opt = torch.optim.Adam(self._actor_critc.parameters(), lr=lr)

        # placeholder for augmentation and intrinsic reward function
        self._dist = None
        self.aug = None
        self.irs = None
    

    def set_encoder(self, encoder):
        # create encoder
        self._encoder = encoder
        self._encoder.train()
        self._encoder_opt = torch.optim.Adam(self._encoder.parameters(), lr=self._lr)
    

    def set_dist(self, dist):
        # create dist function
        self._dist = dist


    def act(self, obs: ndarray, training: bool = True, step: int = 0) -> Tensor:
        obs = torch.as_tensor(obs, device=self._device, dtype=torch.float32)
        encoded_obs = self._encoder(obs)

        mu, values = self._actor_critc(encoded_obs)
        dist = self._dist(mu)

        if not training and self._action_type == 'cont':
            actions = dist.mean()
        elif not training and self._action_type == 'dis':
            actions = dist.mode()
        else:
            actions = dist.sample()

        log_probs = dist.log_probs(actions)
        entropy = dist.entropy().mean()
        
        return actions, values, log_probs, entropy

    def update(self, ):
        pass