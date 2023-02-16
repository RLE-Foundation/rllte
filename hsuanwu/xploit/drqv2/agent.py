from torch.nn import functional as F
from torch import nn
import torch

from hsuanwu.common.typing import *
from hsuanwu.xploit.drqv2.actor import Actor
from hsuanwu.xploit.drqv2.critic import Critic
from hsuanwu.xploit import utils

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
                aug: nn.Module = None,
                noise: Distribution = None,
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

        # create augmentation function
        self.aug = aug()

        # create noise function
        self.noise = noise


    def act(self, obs: ndarray, training: bool = True, step: int = 0) -> Tensor:
        obs = torch.as_tensor(obs, device=self.device)
        encoded_obs = self.encoder(obs.unsequeeze(0))
        mu = self.actor(encoded_obs)
        std = utils.schedule(self.stddev_schedule, step)
        dist = self.noise(loc=mu, scale=torch.ones_like(mu) * std)

        if not training:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0]


    def update(self, batch: Batch, step: int = 0) -> Tensor:
        if step % self.update_every_steps != 0:
            return {}
        
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # obs augmentation
        aug_obs = self.aug(obs.float())
        aug_next_obs = self.aug(next_obs.float())

        # encode
        encoded_obs = self.encoder(aug_obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(aug_next_obs)
        
        # update criitc
        critic_metric = self.update_critic(encoded_obs, action, reward, discount, encoded_next_obs, step)

        # update actor (do not udpate encoder)
        actor_metric = self.update_actor(encoded_obs.detach(), step)

        # udpate critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return {critic_metric, actor_metric}
    

    def update_critic(self, obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs, step: int) -> InfoDict:
        with torch.no_grad():
            mu = self.actor(next_obs)
            std = utils.schedule(self.stddev_schedule, step)
            dist = self.noise(loc=mu, scale=torch.ones_like(mu) * std)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return {'critic_loss': critic_loss, 
                'critic_q1': Q1.mean().item(), 
                'critic_q2': Q2.mean().item(), 
                'critic_target': target_Q.mean().item()}

    def update_actor(self, obs: Tensor, step: int) -> InfoDict:
        mu = self.actor(obs)
        std = utils.schedule(self.stddev_schedule, step)
        dist = self.noise(loc=mu, scale=torch.ones_like(mu) * std)
        action = dist.sample(clip=self.stddev_clip)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = - Q.mean()

         # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return {'actor_loss': actor_loss.item()}