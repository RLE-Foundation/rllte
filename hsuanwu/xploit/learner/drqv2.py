from torch.nn import functional as F
from torch import nn
import torch, time

from hsuanwu.common.typing import *
from hsuanwu.xploit import utils

from hsuanwu.xploit.encoder import VanillaCnnEncoder

class Actor(nn.Module):
    """Actor network

    Args:
        action_space: Action space of the environment.
        features_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.
    
    Returns:
        Actor network.
    """
    def __init__(self, action_space: Space, features_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(features_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(features_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_space.shape[0]))
    
        self.apply(utils.network_init)
    
    def forward(self, obs: Tensor) -> Tensor:
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)

        return mu
    


class Critic(nn.Module):
    """Critic network

    Args:
        action_space: Action space of the environment.
        features_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.
    
    Returns:
        Critic network.
    """
    def __init__(self, action_space: Space, features_dim: int = 64, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(features_dim), nn.Tanh())

        action_shape = action_space.shape
        self.Q1 = nn.Sequential(
            nn.Linear(features_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(features_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.network_init)
    
    def forward(self, obs: Tensor, action: Tensor):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQv2Agent:
    """Learner for continuous control tasks.
        Current learner: DrQ-v2
        Paper: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning
        Link: https://openreview.net/pdf?id=_SJ-_yyes8

    Args:
        obs_space: The observation shape of the environment.
        action_shape: The action shape of the environment.
        feature_dim: Number of features extracted.
        hidden_dim: The size of the hidden layers.
        lr: The learning rate.
        critic_target_tau: The critic Q-function soft-update rate.
        update_every_steps: The agent update frequency.
        num_expl_steps: The exploration steps.
        stddev_schedule: The exploration std schedule.
        stddev_clip: The exploration std clip range.
    
    Returns:
        Agent instance.
    """
    def __init__(self,
                observation_space: Space, 
                action_space: Space,
                device: torch.device = 'cuda',
                feature_dim: int = 50,
                hidden_dim: int = 1024,
                lr: float = 1e-4,
                critic_target_tau: float = 0.01,
                num_expl_steps: int = 2000,
                update_every_steps: int = 2,
                stddev_schedule: str = 'linear(1.0, 0.1, 100000)',
                stddev_clip: float = 0.3) -> None:
        self.device = torch.device(device)
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # create models
        # self.encoder = encoder(
        #     observation_space=observation_space, 
        #     feature_dim=feature_dim)
        # self.encoder = VanillaCnnEncoder(observation_space, features_dim=feature_dim).to(self.device)
        self.encoder = None
        self.actor = Actor(
            action_space=action_space,
            features_dim=feature_dim,
            hidden_dim=hidden_dim).to(self.device)
        self.critic = Critic(
            action_space=action_space,
            features_dim=feature_dim,
            hidden_dim=hidden_dim).to(self.device)
        self.critic_target = Critic(
            action_space=action_space,
            features_dim=feature_dim,
            hidden_dim=hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers
        self.encoder_opt = None # torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target.train()

        # create augmentation function
        self.aug = None

        # create noise function
        self.dist = None

        # create intrinsic reward function
        self.irs = None

    def act(self, obs: ndarray, training: bool = True, step: int = 0) -> Tensor:
        obs = torch.as_tensor(obs, device=self.device)
        encoded_obs = self.encoder(obs.unsqueeze(0))
        # sample actions
        mu = self.actor(encoded_obs)
        std = utils.schedule(self.stddev_schedule, step)
        dist = self.dist(mu=mu, sigma=torch.ones_like(mu) * std)

        if not training:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0]


    def update(self, replay_iter: DataLoader, step: int = 0) -> Tensor:
        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # obs augmentation
        aug_obs = self.aug(obs.float())
        aug_next_obs = self.aug(next_obs.float())

        # encode
        encoded_obs = self.encoder(aug_obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(aug_next_obs)
        
        # update criitc
        metrics.update(
            self.update_critic(encoded_obs, action, reward, discount, encoded_next_obs, step))

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # udpate critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
    

    def update_critic(self, obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs, step: int) -> Dict:
        with torch.no_grad():
            # sample actions
            mu = self.actor(next_obs)
            std = utils.schedule(self.stddev_schedule, step)
            dist = self.dist(mu=mu, sigma=torch.ones_like(mu) * std)
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

        return {'critic_loss': critic_loss.item(), 
                'critic_q1': Q1.mean().item(), 
                'critic_q2': Q2.mean().item(), 
                'critic_target': target_Q.mean().item()}

    def update_actor(self, obs: Tensor, step: int) -> Dict:
        # sample actions
        mu = self.actor(obs)
        std = utils.schedule(self.stddev_schedule, step)
        dist = self.dist(mu=mu, sigma=torch.ones_like(mu) * std)
        action = dist.sample(clip=self.stddev_clip)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = - Q.mean()

         # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return {'actor_loss': actor_loss.item()}