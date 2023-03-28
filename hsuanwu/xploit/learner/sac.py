from torch.nn import functional as F
from torch import nn
import numpy as np
import torch

from hsuanwu.common.typing import *
from hsuanwu.xploit.learner import BaseLearner
from hsuanwu.xploit import utils



class Actor(nn.Module):
    """Actor network.

    Args:
        action_space: Action space of the environment.
        feature_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.
    
    Returns:
        Actor network instance.
    """
    def __init__(self, 
                 action_space: Space, 
                 feature_dim: int = 64, 
                 hidden_dim: int = 1024,
                 log_std_range: Tuple = (-10, 2),
                 ) -> None:
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 2* action_space.shape[0]))
        # placeholder for distribution
        self.dist = None
        self.log_std_min, self.log_std_max = log_std_range
    
        self.apply(utils.network_init)
    

    def forward(self, 
                obs: Tensor, 
                ) -> Tensor:
        """Get actions.

        Args:
            obs: Observations.
        
        Returns:
            Hsuanwu distribution.
        """
        mu, log_std = self.policy(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()
        
        return self.dist(mu, std)
        

class Critic(nn.Module):
    """Critic network.

    Args:
        action_space: Action space of the environment.
        feature_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.
    
    Returns:
        Critic network instance.
    """
    def __init__(self, 
                 action_space: Space, 
                 feature_dim: int = 64, 
                 hidden_dim: int = 1024) -> None:
        super().__init__()

        action_shape = action_space.shape
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, 1))

        self.apply(utils.network_init)
    

    def forward(self, obs: Tensor, action: Tensor):
        """Value estimation.
        
        Args:
            obs: Observations.
            action: Actions.
        
        Returns:
            Estimated values.
        """
        h_action = torch.cat([obs, action], dim=-1)
        
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class SACLearner(BaseLearner):
    """Soft Actor-Critic (SAC) Learner
    
    Args:
        observation_space: Observation space of the environment.
        action_space: Action shape of the environment.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        lr: The learning rate.
        eps: Term added to the denominator to improve numerical stability.

        hidden_dim: The size of the hidden layers.
        critic_target_tau: The critic Q-function soft-update rate.
        update_every_steps: The agent update frequency.
        num_init_steps: The exploration steps.
        log_std_range: Range of std for sampling actions.
        betas: coefficients used for computing running averages of gradient and its square.
        temperature: Initial temperature coefficient.
        fixed_temperature: Fixed temperature or not.
        discount: Discount factor.
    
    Returns:
        Soft Actor-Critic learner instance.
    """
    def __init__(self, 
                 observation_space: Space, 
                 action_space: Space, 
                 action_type: str, 
                 device: torch.device = 'cuda', 
                 feature_dim: int = 5, 
                 lr: float = 1e-4, 
                 eps: float = 0.00008,
                 hidden_dim: int = 1024,
                 critic_target_tau: float = 0.005,
                 num_init_steps: int = 5000,
                 update_every_steps: int = 2,
                 log_std_range: Tuple[float] = (-5., 2),
                 betas: Tuple[float] = (0.9, 0.999),
                 temperature: float = 0.1,
                 fixed_temperature: bool = False,
                 discount: float = 0.99
                 ) -> None:
        super().__init__(observation_space, action_space, action_type, device, feature_dim, lr, eps)

        self._critic_target_tau = critic_target_tau
        self._update_every_steps = update_every_steps
        self._num_init_steps = num_init_steps
        self._fixed_temperature = fixed_temperature
        self._discount = discount

        # create models
        self._encoder = None
        self._actor = Actor(
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            log_std_range=log_std_range).to(self._device)
        self._critic = Critic(
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim).to(self._device)
        self._critic_target = Critic(
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim).to(self._device)
        self._critic_target.load_state_dict(self._critic.state_dict())

        # target entropy
        self._target_entropy = - np.prod(action_space.shape)
        self._log_alpha = torch.tensor(np.log(temperature), device=self._device, requires_grad=True)

        # create optimizers
        self._actor_opt = torch.optim.Adam(self._actor.parameters(), lr=self._lr, betas=betas)
        self._critic_opt = torch.optim.Adam(self._critic.parameters(), lr=self._lr, betas=betas)
        self._log_alpha_opt = torch.optim.Adam([self._log_alpha], lr=self._lr, betas=betas)

        self.train()
        self._critic_target.train()


    def train(self, training=True):
        """ Set the train mode.

        Args:
            training: True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self._actor.train(training)
        self._critic.train(training)
        if self._encoder is not None:
            self._encoder.train(training)
    

    def set_dist(self, dist):
        """Set the distribution for actor.
        
        Args:
            dist: Hsuanwu distribution class.
        
        Returns:
            None.
        """
        self._actor.dist = dist
    

    @property
    def _alpha(self):
        """Get the temperature coefficient.
        """
        return self._log_alpha.exp()

    
    def act(self, obs: ndarray, training: bool = True, step: int = 0) -> Tensor:
        """Make actions based on observations.
        
        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
        obs = torch.as_tensor(obs, device=self._device)
        encoded_obs = self._encoder(obs.unsqueeze(0))
        # sample actions
        dist = self._actor(obs=encoded_obs)

        if not training:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self._num_init_steps:
                action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0]


    def update(self, replay_buffer: Generator, step: int = 0) -> Dict:
        """Update the learner.
        
        Args:
            replay_buffer: Hsuanwu replay buffer.
            step: Global training step.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """
        metrics = {}
        if step % self._update_every_steps != 0:
            return metrics
        
        obs, action, reward, done, next_obs = replay_buffer.sample()

        if self._irs is not None:
            intrinsic_reward = self._irs.compute_irs(
                rollouts={'observations': obs.unsqueeze(1).numpy(), 
                          'actions': action.unsqueeze(1).numpy()},
                step=step)
            reward += torch.as_tensor(intrinsic_reward, dtype=torch.float32).squeeze(1)
        

        # obs augmentation
        if self._aug is not None:
            obs = self._aug(obs.float())
            next_obs = self._aug(next_obs.float())

        # encode
        encoded_obs = self._encoder(obs)
        with torch.no_grad():
            encoded_next_obs = self._encoder(next_obs)

        # update criitc
        metrics.update(
            self.update_critic(encoded_obs, 
                               action, 
                               reward, 
                               done,
                               encoded_next_obs, step))

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor_and_alpha(encoded_obs.detach(), step))

        # udpate critic target
        utils.soft_update_params(self._critic, self._critic_target, self._critic_target_tau)

        return metrics


    def update_critic(self, 
                      obs: Tensor, 
                      action: Tensor, 
                      reward: Tensor, 
                      done: Tensor,
                      next_obs: Tensor, 
                      step: int) -> Dict:
        """Update the critic network.
        
        Args:
            obs: Observations.
            action: Actions.
            reward: Rewards.
            done: Dones.
            next_obs: Next observations.
            step: Global training step.
        
        Returns:
            Critic loss metrics.
        """
        with torch.no_grad():
            dist = self._actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self._critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self._alpha.detach() * log_prob
            target_Q = reward + (1. - done) * self._discount * target_V
        
        Q1, Q2 = self._critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self._encoder_opt.zero_grad(set_to_none=True)
        self._critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._critic_opt.step()
        self._encoder_opt.step()

        return {'critic_loss': critic_loss.item(), 
                'critic_q1': Q1.mean().item(), 
                'critic_q2': Q2.mean().item(), 
                'critic_target': target_Q.mean().item()}


    def update_actor_and_alpha(self, obs: Tensor, step: int) -> Dict:
        """Update the actor network and temperature.
        
        Args:
            obs: Observations.
            step: Global training step.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self._actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self._critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = (self._alpha.detach() * log_prob - Q).mean()

         # optimize actor
        self._actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._actor_opt.step()

        if not self._fixed_temperature:
            # update temperature
            self._log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss = (self._alpha * (-log_prob - self._target_entropy).detach()).mean()
            alpha_loss.backward()
            self._log_alpha_opt.step()
        else:
            alpha_loss = torch.scalar_tensor(s=0.0)

        return {'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item()}