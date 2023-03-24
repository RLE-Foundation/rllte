from torch.nn import functional as F
from torch import nn
import numpy as np
import torch

from hsuanwu.common.typing import *
from hsuanwu.xploit.learner import BaseLearner
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
        
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Tanh(),
                                   nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)
        self.aux_critic = nn.Linear(hidden_dim, 1)
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)
    

    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        return self.critic(self.trunk(obs))


    def get_action(self, obs: Tensor) -> Tensor:
        """Get deterministic actions for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        logits = self.actor(self.trunk(obs))
        return self.dist(logits).mean


    def get_action_and_value(self, obs: Tensor, actions: Tensor = None) -> Sequence[Tensor]:
        """Get actions and estimated values for observations.
        
        Args:
            obs: Sampled observations.
            actions: Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.trunk(obs)
        logits = self.actor(h)
        dist = self.dist(logits)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_probs(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy
    
    
    def get_probs_and_aux_value(self, obs: Tensor) -> Sequence[Tensor]:
        """Get probs and auxiliary estimated values for auxiliary phase update.
        
        Args:
            obs: Sampled observations.
        
        Returns:
            Distribution, estimated values, auxiliary estimated values.
        """
        h = self.trunk(obs)
        logits = self.actor(h)
        dist = self.dist(logits)

        return dist, self.critic(h.detach()), self.aux_critic(h)


    def get_logits(self, obs: Tensor) -> Distribution:
        """Get the log-odds of sampling.

        Args:
            obs: Sampled observations.
        
        Returns:
            Distribution
        """
        return self.dist(self.actor(self.trunk(obs)))


class PPGLearner(BaseLearner):
    """Phasic Policy Gradient (PPG) Learner.
    
    Args:
        observation_space: Observation space of the environment.
        action_space: Action space of the environment.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        lr: The learning rate.
        eps: Term added to the denominator to improve numerical stability.

        hidden_dim: The size of the hidden layers.
        clip_range: Clipping parameter.
        num_policy_mini_batch: Number of mini-batches in policy phase.
        num_aux_mini_batch: Number of mini-batches in auxiliary phase.
        vf_coef: Weighting coefficient of value loss.
        ent_coef: Weighting coefficient of entropy bonus.
        max_grad_norm: Maximum norm of gradients.
        policy_epochs: Number of iterations in the policy phase.
        aux_epochs: Number of iterations in the auxiliary phase.
        kl_coef: Weighting coefficient of divergence loss.
        num_aux_grad_accum: Number of gradient accumulation for auxiliary phase update.

    Returns:
        PPG learner instance.
    """
    def __init__(self, 
                 observation_space: Space, 
                 action_space: Space, 
                 action_type: str, 
                 device: torch.device = 'cuda', 
                 feature_dim: int = 256,
                 lr: float = 5e-4,
                 eps: float = 1e-5,
                 hidden_dim: int = 256,
                 clip_range: float = 0.2,
                 num_policy_mini_batch: int = 8,
                 num_aux_mini_batch: int = 4,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 policy_epochs: int = 32,
                 aux_epochs: int = 6,
                 kl_coef: float = 1.0,
                 num_aux_grad_accum: int = 1,
                 ) -> None:
        super().__init__(observation_space, action_space, action_type, device, feature_dim, lr, eps)
        
        self._clip_range = clip_range
        self._num_policy_mini_batch = num_policy_mini_batch
        self._num_aux_mini_batch = num_aux_mini_batch
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self._max_grad_norm = max_grad_norm
        self._policy_epochs = policy_epochs
        self._aux_epochs = aux_epochs
        self._kl_coef = kl_coef
        self._num_aux_grad_accum = num_aux_grad_accum

        # auxiliary storage
        self._aux_obs = None
        self._aux_returns = None
        self._aux_logits = None

        # create models
        self._encoder = None
        self._ac = ActorCritic(
            action_space=self._action_space,
            feature_dim=self._feature_dim,
            hidden_dim=hidden_dim
        ).to(self._device)

        self._ac_opt = torch.optim.Adam(self._ac.parameters(), lr=lr, eps=eps)
        self.train()
    
    
    def train(self, training=True) -> None:
        """ Set the train mode.
        """
        self.training = training
        self._ac.train(training)
        if self._encoder is not None:
            self._encoder.train(training)

    
    def set_dist(self, dist: Distribution) -> None:
        """Set the distribution for actor.
        
        Args:
            dist: Hsuanwu distribution class.
        
        Returns:
            None.
        """
        self._dist = dist
        self._ac.dist = dist


    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        encoded_obs = self._encoder(obs)
        return self._ac.get_value(obs=encoded_obs)


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

        if training:
            actions, values, log_probs, entropy = self._ac.get_action_and_value(obs=encoded_obs)
            return actions, values, log_probs, entropy
        else:
            actions = self._ac.get_action(obs=encoded_obs)
            return actions
    

    def update(self, rollout_buffer: Any, episode: int = 0) -> Dict:
        """Update learner.
        
        Args:
            rollout_buffer: Hsuanwu rollout buffer.
            episode: Global training episode.
        
        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """

        # TODO: Save auxiliary transitions
        if episode == 0:
            num_steps, num_envs = rollout_buffer.obs.size()[:2]
            self._aux_obs = torch.empty(
                size=(num_steps, num_envs*self._policy_epochs, *self._obs_space.shape), 
                device='cpu', dtype=torch.float32)
            self._aux_returns = torch.empty(
                size=(num_steps, num_envs*self._policy_epochs, 1), 
                device='cpu', dtype=torch.float32)
            self._aux_logits = torch.empty(
                size=(num_steps, num_envs*self._policy_epochs, self._action_space.shape[0]),
                device='cpu', dtype=torch.float32)
            self._num_aux_rollouts = num_envs * self._policy_epochs
            self._num_envs = num_envs
            self._num_steps = num_steps
            
        idx = int(episode % self._policy_epochs)
        self._aux_obs[:, idx*self._num_envs:(idx+1)*self._num_envs].copy_(rollout_buffer.obs.clone())
        self._aux_returns[:, idx*self._num_envs:(idx+1)*self._num_envs].copy_(rollout_buffer.returns.clone())


        # TODO: Policy phase
        total_actor_loss = 0.
        total_critic_loss = 0.
        total_entropy_loss = 0.
        num_updates = 0

        generator = rollout_buffer.generator(self._num_policy_mini_batch)

        for batch in generator:
            batch_obs, batch_actions, batch_values, batch_returns, \
                batch_dones, batch_old_log_probs, adv_targ = batch
                
            # evaluate sampled actions
            _, values, log_probs, entropy = self._ac.get_action_and_value(
                obs=self._encoder(batch_obs),
                actions=batch_actions)

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
            num_updates += 1
        

        total_actor_loss /= num_updates
        total_critic_loss /= num_updates
        total_entropy_loss /= num_updates


        if  (episode + 1) % self._policy_epochs != 0:
            # if not auxiliary phase, return train loss directly.
            return {
                'actor_loss': total_actor_loss,
                'critic_loss': total_critic_loss,
                'entropy_loss': total_entropy_loss
                }



        # TODO: Auxiliary phase
        for idx in range(self._policy_epochs):
            with torch.no_grad():
                aux_obs = self._aux_obs[:, idx*self._num_envs:(idx+1)*self._num_envs].to(self._device).reshape(
                    -1, *self._aux_obs.size()[2:])
                # get logits
                logits = self._ac.get_logits(self._encoder(aux_obs)).logits.cpu().clone()
                self._aux_logits[:, idx*self._num_envs:(idx+1)*self._num_envs] = logits.reshape(
                    self._num_steps, self._num_envs, self._aux_logits.size()[2])


        for e in range(self._aux_epochs):
            print('Auxiliary Phase', e)
            aux_inds = np.arange(self._num_aux_rollouts)
            np.random.shuffle(aux_inds)

            for idx in range(0, self._num_aux_rollouts, self._num_aux_mini_batch):
                batch_inds = aux_inds[idx:idx+self._num_aux_mini_batch]
                batch_aux_obs = self._aux_obs[:, batch_inds].reshape(-1, *self._aux_obs.size()[2:]).to(self._device)
                batch_aux_returns = self._aux_returns[:, batch_inds].reshape(-1, *self._aux_returns.size()[2:]).to(self._device)
                batch_aux_logits = self._aux_logits[:, batch_inds].reshape(-1, *self._aux_logits.size()[2:]).to(self._device)

                new_dist, new_values, new_aux_values = self._ac.get_probs_and_aux_value(
                    self._encoder(batch_aux_obs))
                
                new_values = new_values.view(-1)
                new_aux_values = new_aux_values.view(-1)
                old_dist = self._dist(logits=batch_aux_logits)
                # divergence loss
                kl_loss = torch.distributions.kl_divergence(old_dist, new_dist).mean()
                # value loss
                value_loss = 0.5 * ((new_values - batch_aux_returns)).mean()
                aux_value_loss = 0.5 * ((new_aux_values - batch_aux_returns)).mean()
                # total loss
                (value_loss + aux_value_loss + self._kl_coef * kl_loss).backward()

                if (idx + 1) % self._num_aux_grad_accum == 0:
                    self._encoder_opt.zero_grad(set_to_none=True)
                    self._ac_opt.zero_grad(set_to_none=True)
                    nn.utils.clip_grad_norm_(self._encoder.parameters(), self._max_grad_norm)
                    nn.utils.clip_grad_norm_(self._ac.parameters(), self._max_grad_norm)
                    self._encoder_opt.step()
                    self._ac_opt.step()



                
