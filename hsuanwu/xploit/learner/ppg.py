from torch.nn import functional as F
from torch import nn
import torch

from hsuanwu.common.typing import *
from hsuanwu.xploit.learner.base import BaseLearner
from hsuanwu.xploit import utils


class ActorCritic(nn.Module):
    """Actor-Critic network.
    
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
        self.aux_critic = nn.Linear(hidden_dim, 1)

        self.apply(utils.network_init)
    
    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.trunk(obs))


    def get_action_and_value(self, obs: Tensor, actions: Tensor, dist: Distribution) -> Sequence[Tensor]:
        """Get actions and estimated values for observations.
        
        Args:
            obs: Sampled observations.
            actions: Sampled actions.
            dist: Hsuanwu distribution class.

        Returns:
            Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        encoded_obs = self._encoder(obs)
        mu, values = self._ac(encoded_obs)
        d = dist(mu)

        log_probs = d.log_probs(actions)
        entropy = d.entropy().mean()

        return values, log_probs, entropy
    

class PPGLearner(BaseLearner):
    """Phasic Policy Gradient (PPG) Learner.
    
    Args:
        observation_space: Observation space of the environment.
        action_space: Action space of the environment.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        lr: The learning rate.

        hidden_dim: The size of the hidden layers.
        eps: RMSprop optimizer epsilon.
        clip_range: Clipping parameter.
        num_mini_batch: Number of mini-batches.
        vf_coef: Weighting coefficient of value loss.
        ent_coef: Weighting coefficient of entropy bonus.
        max_grad_norm: Maximum norm of gradients.
        policy_epochs: Number of iterations in the policy phase.
        auxiliary_epochs: Number of iterations in the auxiliary phase.

    Returns:
        PPG learner instance.
    """
    def __init__(self, 
                 observation_space: Space, 
                 action_space: Space, 
                 action_type: str, 
                 device: torch.device = 'cuda', 
                 feature_dim: int = 50,
                 lr: float = 2.5e-4,
                 hidden_dim: int = 256,
                 eps: float = 1e-5,
                 clip_range: float = 0.2,
                 num_mini_batch: int = 4,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 policy_epochs: int = 32,
                 auxiliary_epochs: int = 6
                 ) -> None:
        super().__init__(observation_space, action_space, action_type, device, feature_dim, lr)
        self._eps = eps
        self._clip_range = clip_range
        self._num_mini_batch = num_mini_batch
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self._max_grad_norm = max_grad_norm
        self._policy_epochs = policy_epochs
        self._auxiliary_epochs = auxiliary_epochs

        # auxiliary storage
        self._aux_obs = None
        self._aux_returns = None
        self._aux_probs = None
        self._first_episode = True

        # create models
        self._encoder = None
        self._ac = ActorCritic(
            action_space=self._obs_space,
            feature_dim=self._feature_dim,
            hidden_dim=hidden_dim
        ).to(self._device)

        self._ac_opt = torch.optim.Adam(self._ac.parameters(), lr=lr, eps=eps)
        self.train()

        # placeholder for augmentation and intrinsic reward function
        self._dist = None
        self._aug = None
        self._irs = None
    
    
    def train(self, training=True) -> None:
        """ Set the train mode.
        """
        self.training = training
        self._ac.train(training)
        if self._encoder is not None:
            self._encoder.train(training)
    

    def act(self, obs: Tensor, training: bool = True, step: int = 0) -> Tensor:
        """Sample actions.
        
        Args:
            obs: Observation tensor.
            training: Training or testing.
            step: Global training step.
        
        Returns:
            Sampled actions.
        """
        encoded_obs = self._encoder(obs)
        mu, values = self._ac(encoded_obs)
        dist = self._dist(mu)

        if not training:
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
        features = self._ac.trunk(encoded_obs)
        value = self._ac.critic(features)
        
        return value
    

    def get_probs(self, obs: Tensor) -> Tensor:
        """Get action probabilities.
        
        Args:
            obs: Sampled observations.
        
        Returns:
            Probabilities.
        """
        encoded_obs = self._encoder(obs)
        features = self._ac.trunk(encoded_obs)
        mu = self._ac.actor(features)
        dist = self._dist(mu)

        return dist.probs()
    
    
    def get_aux_value(self, obs: Tensor) -> Tensor:
        """Compute auxiliary estimated values of observations.
        
        Args:
            obs: Observations.
        
        Returns:
            Estimated values.
        """
        encoded_obs = self._encoder(obs)
        features = self._ac.trunk(encoded_obs)
        aux_value = self._ac.aux_critic(features)
        
        return aux_value
    

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Sequence[Tensor]:
        """Evaluate sampled actions.
        
        Args:
            obs: Sampled observations.
            actions: Samples actions.
            aux: Get auxiliary values?

        Returns:
            Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        encoded_obs = self._encoder(obs)
        mu, values = self._ac(encoded_obs)
        dist = self._dist(mu)

        log_probs = dist.log_probs(actions)
        entropy = dist.entropy().mean()

        return values, log_probs, entropy


    def update(self, rollout_buffer: Any, episode: int = 0) -> Dict:
        """Update learner.
        
        Args:
            rollout_buffer: Hsuanwu rollout buffer.
            step: Global training step.
        
        Returns:
            Training metrics such as loss functions.
        """

        # TODO: Policy phase
        total_actor_loss = 0.
        total_critic_loss = 0.
        total_entropy_loss = 0.

        if episode % self._policy_epochs != 0:
            generator = rollout_buffer.generator(self._num_mini_batch)

            if self._first_episode:
                num_steps, num_envs = rollout_buffer.obs.size()[:2]
                self._aux_obs = torch.empty(
                    size=(num_steps, num_envs*self._policy_epochs)+self._obs_space.shape, 
                    device=self._device, dtype=torch.float32)
                self._aux_returns = torch.empty(
                    size=(num_steps, num_envs*self._policy_epochs, 1), 
                    device=self._device, dtype=torch.float32)
                self._aux_probs = torch.empty(
                    size=(num_steps, num_envs*self._policy_epochs)+self._action_space.shape[0],
                    device=self._device, dtype=torch.float32)
            else:
                idx = int(episode % self._policy_epochs)
                self._aux_obs[:, idx] = rollout_buffer.obs.copy()
                self._aux_returns[:, idx] = rollout_buffer.returns.copy()

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
                self._ac_opt.zero_grad(set_to_none=True)
                (critic_loss * self._vf_coef + actor_loss - entropy * self._ent_coef).backward()
                nn.utils.clip_grad_norm_(self._encoder.parameters(), self._max_grad_norm)
                nn.utils.clip_grad_norm_(self._ac.parameters(), self._max_grad_norm)
                self._encoder_opt.step()
                self._ac_opt.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()

            return {
                'actor_loss': total_actor_loss,
                'critic_loss': total_critic_loss,
                'entropy_loss': total_entropy_loss
            }
        

        # TODO: Get action probs for stored auxiliary observations.
        for idx in range(self._policy_epochs):
            with torch.no_grad():
                probs = self.get_probs(self._aux_obs[:, idx])
                self._aux_probs[:, idx] = probs
        

        # TODO: Auxiliary phase update
        if episode % self._auxiliary_epochs == 0:
            pass