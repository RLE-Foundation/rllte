from torch.nn import functional as F
from torch import nn
import torch

from hsuanwu.common.typing import *
from hsuanwu.xploit.drqv2.actor import Actor
from hsuanwu.xploit.drqv2.critic import Critic

class DrQv2Agent:
    """
    Learner for continuous control tasks.
    Current learner: DrQ-v2
    Paper: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning
    Link: https://openreview.net/pdf?id=_SJ-_yyes8

    :param obs_space: The observation shape of the environment.
    :param action_shape: The action shape of the environment.
    :param feature_dim: Number of features extracted.
    :param hidden_dim: The size of the hidden layers.
    :param lr: The learning rate.
    :param critic_target_tau: The critic Q-function soft-update rate.
    :param update_every_steps: The agent update frequency.
    :param num_expl_steps: The exploration steps.
    :param stddev_schedule: The exploration std schedule.
    :param stddev_clip: The exploration std clip range.
    """
    def __init__(self,
                observation_space: Space, 
                action_space: Space,
                device: torch.device = 'cuda',
                encoder: nn.Module = None,
                feature_dim: int = 50,
                hidden_dim: int = 1024,
                lr: float = 1e-4,
                critic_target_tau: float = 0.01,
                num_expl_steps: int = 2000,
                update_every_steps: int = 2,
                stddev_schedule: str = 'linear(1.0, 0.1, 100000)',
                stddev_clip: float = 0.3) -> None:
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # create models
        self.encoder = encoder(
            observation_space=observation_space, 
            feature_dim=feature_dim)
        self.actor = Actor(
            action_space=action_space,
            features_dim=feature_dim,
            hidden_dim=hidden_dim).to(device)
        self.critic = Critic(
            action_space=action_space,
            features_dim=feature_dim,
            hidden_dim=hidden_dim).to(device)
        self.critic_target = Critic(
            action_space=action_space,
            features_dim=feature_dim,
            hidden_dim=hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, obs, training=True):
        pass
    
    def update(self, batch: Batch):
        pass
