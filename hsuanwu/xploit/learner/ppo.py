import torch
from torch import nn
from torch.nn import functional as F

from hsuanwu.common.typing import *
from hsuanwu.xploit import utils
from hsuanwu.xploit.learner import BaseLearner


class ActorCritic(nn.Module):
    """Actor-Critic network.

    Args:
        action_space: Action space of the environment.
        feature_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.

    Returns:
        Actor-Critic instance.
    """

    def __init__(self, action_space: Space, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_space.shape[0])
        self.critic = nn.Linear(hidden_dim, 1)
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
        mu = self.actor(self.trunk(obs))
        return self.dist(mu).mean

    def get_action_and_value(
        self, obs: Tensor, actions: Tensor = None
    ) -> Sequence[Tensor]:
        """Get actions and estimated values for observations.

        Args:
            obs: Sampled observations.
            actions: Sampled actions.

        Returns:
            Actions, Estimated values, log of the probability evaluated at `actions`, entropy of distribution.
        """
        h = self.trunk(obs)
        mu = self.actor(h)
        dist = self.dist(mu)
        if actions is None:
            actions = dist.sample()

        log_probs = dist.log_probs(actions)
        entropy = dist.entropy().mean()

        return actions, self.critic(h), log_probs, entropy


class PPOLearner(BaseLearner):
    """Proximal Policy Optimization (PPO) Learner.

    Args:
        observation_space: Observation space of the environment.
        action_space: Action shape of the environment.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        lr: The learning rate.
        eps: Term added to the denominator to improve numerical stability.

        hidden_dim: The size of the hidden layers.
        clip_range: Clipping parameter.
        n_epochs: Times of updating the policy.
        num_mini_batch: Number of mini-batches.
        vf_coef: Weighting coefficient of value loss.
        ent_coef: Weighting coefficient of entropy bonus.
        max_grad_norm: Maximum norm of gradients.

    Returns:
        PPO learner instance.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: torch.device = "cuda",
        feature_dim: int = 256,
        lr: float = 5e-4,
        eps: float = 1e-5,
        hidden_dim: int = 256,
        clip_range: float = 0.2,
        n_epochs: int = 3,
        num_mini_batch: int = 8,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        super().__init__(
            observation_space, action_space, action_type, device, feature_dim, lr, eps
        )

        self._n_epochs = n_epochs
        self._clip_range = clip_range
        self._num_mini_batch = num_mini_batch
        self._vf_coef = vf_coef
        self._ent_coef = ent_coef
        self._max_grad_norm = max_grad_norm

        # create models
        self._encoder = None
        self._ac = ActorCritic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self._device)

        # create optimizers
        self._ac_opt = torch.optim.Adam(self._ac.parameters(), lr=lr, eps=eps)
        self.train()

    def train(self, training=True):
        """Set the train mode.

        Args:
            training: True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self._ac.train(training)
        if self._encoder is not None:
            self._encoder.train(training)

    def set_dist(self, dist):
        """Set the distribution for actor.

        Args:
            dist: Hsuanwu distribution class.

        Returns:
            None.
        """
        self._dist = dist
        self._ac.dist = dist

    def act(
        self, obs: ndarray, training: bool = True, step: int = 0
    ) -> Sequence[Tensor]:
        """Make actions based on observations.

        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self._encoder(obs)

        if training:
            actions, values, log_probs, entropy = self._ac.get_action_and_value(
                obs=encoded_obs
            )
            return actions, values, log_probs, entropy
        else:
            actions = self._ac.get_action(obs=encoded_obs)
            return actions

    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs: Observations.

        Returns:
            Estimated values.
        """
        encoded_obs = self._encoder(obs)
        return self._ac.get_value(obs=encoded_obs)

    def update(self, rollout_buffer: Any, episode: int = 0) -> Dict:
        """Update the learner.

        Args:
            rollout_buffer: Hsuanwu rollout buffer.
            episode: Global training episode.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0

        for e in range(self._n_epochs):
            generator = rollout_buffer.generator(self._num_mini_batch)

            for batch in generator:
                (
                    batch_obs,
                    batch_actions,
                    batch_values,
                    batch_returns,
                    batch_dones,
                    batch_old_log_probs,
                    adv_targ,
                ) = batch

                # evaluate sampled actions
                _, values, log_probs, entropy = self._ac.get_action_and_value(
                    obs=self._encoder(batch_obs), actions=batch_actions
                )

                # actor loss part
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - self._clip_range, 1.0 + self._clip_range)
                    * adv_targ
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss part
                values_clipped = batch_values + (values - batch_values).clamp(
                    -self._clip_range, self._clip_range
                )
                values_losses = (batch_values - batch_returns).pow(2)
                values_losses_clipped = (values_clipped - batch_returns).pow(2)
                critic_loss = (
                    0.5 * torch.max(values_losses, values_losses_clipped).mean()
                )

                # update
                self._encoder_opt.zero_grad(set_to_none=True)
                self._ac_opt.zero_grad(set_to_none=True)
                (
                    critic_loss * self._vf_coef + actor_loss - entropy * self._ent_coef
                ).backward()
                nn.utils.clip_grad_norm_(
                    self._encoder.parameters(), self._max_grad_norm
                )
                nn.utils.clip_grad_norm_(self._ac.parameters(), self._max_grad_norm)
                self._ac_opt.step()
                self._encoder_opt.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()

        num_updates = self._n_epochs * self._num_mini_batch

        total_actor_loss /= num_updates
        total_critic_loss /= num_updates
        total_entropy_loss /= num_updates

        return {
            "actor_loss": total_actor_loss,
            "critic_loss": total_critic_loss,
            "entropy": total_entropy_loss,
        }
