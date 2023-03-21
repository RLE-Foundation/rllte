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
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)

        self.apply(utils.network_init)

    
    def forward(self, obs: Tensor) -> Sequence[Tensor]:
        features = self.trunk(obs)
        mu = self.actor(features)
        value = self.critic(features)

        return mu, value


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent
        Paper: Proximal policy optimization algorithms
        Link: https://arxiv.org/pdf/1707.06347
    
    Args:
        observation_space: Observation space of the environment.
        action_space: Action shape of the environment.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        hidden_dim: The size of the hidden layers.
        lr: The learning rate.
        eps: RMSprop optimizer epsilon.
        clip_range: Clipping parameter.
        n_epochs: Times of updating the policy.
        num_mini_batch: Number of mini-batches.
        vf_coef: Weighting coefficient of value loss.
        ent_coef: Weighting coefficient of entropy bonus.
        max_grad_norm: Maximum norm of gradients.
    
    Returns:
        Agent instance.
    """
    def __init__(self,
                 observation_space: Space, 
                 action_space: Space,
                 action_type: str,
                 device: torch.device = 'cuda',
                 feature_dim: int = 256,
                 hidden_dim: int = 1024,
                 lr: float = 1e-4,
                 eps: float = 1e-5,
                 clip_range: float = 0.2,
                 n_epochs: int = 5,
                 num_mini_batch: int = 4,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ) -> None:
        self._action_type = action_type
        self._device = torch.device(device)
        self._lr = lr
        self._eps = eps
        self._n_epochs = n_epochs
        self._clip_range = clip_range
        self._num_mini_batch = num_mini_batch
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self._max_grad_norm = max_grad_norm

        # create models
        self._encoder = None
        self._actor_critic = ActorCritic(
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        ).to(self._device)

        # create optimizers
        self._actor_critic_opt = torch.optim.Adam(self._actor_critic.parameters(), lr=lr, eps=eps)
        self.train()

        # placeholder for augmentation and intrinsic reward function
        self._dist = None
        self._aug = None
        self._irs = None
    
    def train(self, training=True):
        self.training = training
        self._actor_critic.train(training)
        if self._encoder is not None:
            self._encoder.train(training)

    def set_encoder(self, encoder) -> None:
        """Set encoder.
        
        Args:
            encoder: Hsuanwu xploit.encoder instance.
        
        Returns:
            None.
        """
        # create encoder
        self._encoder = encoder
        self._encoder.train()
        self._encoder_opt = torch.optim.Adam(self._encoder.parameters(), lr=self._lr, eps=self._eps)
    

    def set_dist(self, dist):
        """Set distribution for sampling actions
        
        Args:
            dist: Hsuanwu xplore.distribution class.
        
        Returns:
            None.
        """
        self._dist = dist


    def act(self, obs: ndarray, training: bool = True, step: int = 0) -> Sequence[Tensor]:
        """Make actions based on observations.
        
        Args:
            obs: Observations.
            training: training mode, True or False.
            step: global training step.

        Returns:
            Actions.
        """
        encoded_obs = self._encoder(obs)
        mu, values = self._actor_critic(encoded_obs)
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
    

    def get_value(self, obs: Tensor) -> Tensor:
        """Compute estimated values of observations.
        
        Args:
            obs: Observations.
        
        Returns:
            Estimated values.
        """
        encoded_obs = self._encoder(obs)
        features = self._actor_critic.trunk(encoded_obs)
        value = self._actor_critic.critic(features)
        
        return value
    

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Sequence[Tensor]:
        """Evaluate sampled actions.
        
        Args:
            obs: Sampled observations.
            actions: Samples actions.

        Returns:
            Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        encoded_obs = self._encoder(obs)
        mu, values = self._actor_critic(encoded_obs)
        dist = self._dist(mu)

        log_probs = dist.log_probs(actions)
        entropy = dist.entropy().mean()

        return values, log_probs, entropy
    

    def update(self, rollout_buffer: Generator, step: int = 0) -> Dict:
        """Update the learner.
        
        Args:
            rollout_iter: Rollout buffer.
            step: Global training step.

        Returns:
            Training metrics that includes actor loss, critic_loss, etc.
        """
        actor_loss_epoch = 0.
        critic_loss_epoch = 0.
        entropy_loss_epoch = 0.

        for e in range(self._n_epochs):
            generator = rollout_buffer.generator(self._num_mini_batch)

            for batch in generator:
                batch_obs, batch_actions, batch_values, batch_returns, \
                    batch_dones, batch_old_log_probs, adv_targ = batch
                
                # evaluate sampled actions
                values, log_probs, entropy = self.evaluate_actions(batch_obs, batch_actions)

                # actor loss part
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range) * adv_targ
                actor_loss = - torch.min(surr1, surr2).mean()

                # critic loss part
                values_clipped = batch_values + (values - batch_values).clamp(-self._clip_range, self._clip_range)
                values_losses = (batch_values - batch_returns).pow(2)
                values_losses_clipped = (values_clipped - batch_returns).pow(2)
                critic_loss = 0.5 * torch.max(values_losses, values_losses_clipped).mean()

                # update
                self._encoder_opt.zero_grad(set_to_none=True)
                self._actor_critic_opt.zero_grad(set_to_none=True)
                (critic_loss * self._vf_coef + actor_loss - entropy * self._ent_coef).backward()
                nn.utils.clip_grad_norm_(self._encoder.parameters(), self._max_grad_norm)
                nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._max_grad_norm)
                self._encoder_opt.step()
                self._actor_critic_opt.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_loss_epoch += entropy.item()
        
        num_updates = self._n_epochs * self._num_mini_batch


        actor_loss_epoch /= num_updates
        critic_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates


        return {'actor_loss': actor_loss_epoch,
                'critic_loss': critic_loss_epoch,
                'entropy': entropy_loss_epoch}
