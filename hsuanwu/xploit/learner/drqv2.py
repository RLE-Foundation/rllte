import torch
from torch import nn
from torch.nn import functional as F

from hsuanwu.common.typing import *
from hsuanwu.xploit import utils
from hsuanwu.xploit.learner import BaseLearner


class Actor(nn.Module):
    """Actor network.

    Args:
        action_space: Action space of the environment.
        feature_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.

    Returns:
        Actor network instance.
    """

    def __init__(
        self, action_space: Space, feature_dim: int = 64, hidden_dim: int = 1024
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Tanh())
        # self.trunk = nn.Sequential(nn.Linear(32 * 35 * 35, feature_dim),
        #                            nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_space.shape[0]),
        )
        # placeholder for distribution
        self.dist = None

        self.apply(utils.network_init)

    def forward(self, obs: Tensor, std: float = None) -> Tensor:
        """Get actions.

        Args:
            obs: Observations.
            std: Standard deviation for sampling actions.

        Returns:
            Hsuanwu distribution.
        """
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)

        return self.dist(mu, torch.ones_like(mu) * std)


class Critic(nn.Module):
    """Critic network.

    Args:
        action_space: Action space of the environment.
        feature_dim: Number of features accepted.
        hidden_dim: Number of units per hidden layer.

    Returns:
        Critic network instance.
    """

    def __init__(
        self, action_space: Space, feature_dim: int = 64, hidden_dim: int = 1024
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(feature_dim), nn.Tanh())
        # self.trunk = nn.Sequential(nn.Linear(32 * 35 * 35, feature_dim),
        #    nn.LayerNorm(feature_dim), nn.Tanh())

        action_shape = action_space.shape
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.network_init)

    def forward(self, obs: Tensor, action: Tensor):
        """Value estimation.

        Args:
            obs: Observations.
            action: Actions.

        Returns:
            Estimated values.
        """
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQv2Learner(BaseLearner):
    """Data Regularized-Q v2 (DrQ-v2).

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
        stddev_schedule: The exploration std schedule.
        stddev_clip: The exploration std clip range.

    Returns:
        DrQv2 learner instance.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: torch.device = "cuda",
        feature_dim: int = 50,
        lr: float = 1e-4,
        eps: float = 0.00008,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.01,
        num_init_steps: int = 2000,
        update_every_steps: int = 2,
        stddev_schedule: str = "linear(1.0, 0.1, 100000)",
        stddev_clip: float = 0.3,
    ) -> None:
        super().__init__(
            observation_space, action_space, action_type, device, feature_dim, lr, eps
        )

        self._critic_target_tau = critic_target_tau
        self._update_every_steps = update_every_steps
        self._num_init_steps = num_init_steps
        self._stddev_schedule = stddev_schedule
        self._stddev_clip = stddev_clip

        # create models
        self._encoder = None
        self._actor = Actor(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self._device)
        self._critic = Critic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self._device)
        self._critic_target = Critic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self._device)
        self._critic_target.load_state_dict(self._critic.state_dict())

        # create optimizers
        self._actor_opt = torch.optim.Adam(self._actor.parameters(), lr=self._lr)
        self._critic_opt = torch.optim.Adam(self._critic.parameters(), lr=self._lr)
        self.train()
        self._critic_target.train()

    def train(self, training=True):
        """Set the train mode.

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
        std = utils.schedule(self._stddev_schedule, step)
        dist = self._actor(obs=encoded_obs, std=std)

        if not training:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self._num_init_steps:
                action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0]

    def update(self, replay_buffer: DataLoader, step: int = 0) -> Dict:
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

        # batch = next(replay_iter)
        obs, action, reward, discount, next_obs = next(
            replay_buffer
        )  # utils.to_torch(batch, self.device)
        if self._irs is not None:
            intrinsic_reward = self._irs.compute_irs(
                rollouts={
                    "observations": obs.unsqueeze(1).numpy(),
                    "actions": action.unsqueeze(1).numpy(),
                },
                step=step,
            )
            reward += torch.as_tensor(intrinsic_reward, dtype=torch.float32).squeeze(1)

        obs = obs.float().to(self._device)
        action = action.float().to(self._device)
        reward = reward.float().to(self._device)
        discount = discount.float().to(self._device)
        next_obs = next_obs.float().to(self._device)

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
            self.update_critic(
                encoded_obs, action, reward, discount, encoded_next_obs, step
            )
        )

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # udpate critic target
        utils.soft_update_params(
            self._critic, self._critic_target, self._critic_target_tau
        )

        return metrics

    def update_critic(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        discount: Tensor,
        next_obs: Tensor,
        step: int,
    ) -> Dict:
        """Update the critic network.

        Args:
            obs: Observations.
            action: Actions.
            reward: Rewards.
            discount: discounts.
            next_obs: Next observations.
            step: Global training step.

        Returns:
            Critic loss metrics.
        """

        with torch.no_grad():
            # sample actions
            std = utils.schedule(self._stddev_schedule, step)
            dist = self._actor(next_obs, std)

            next_action = dist.sample(clip=self._stddev_clip)
            target_Q1, target_Q2 = self._critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self._critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        # optimize encoder and critic
        self._encoder_opt.zero_grad(set_to_none=True)
        self._critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._critic_opt.step()
        self._encoder_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "critic_q1": Q1.mean().item(),
            "critic_q2": Q2.mean().item(),
            "critic_target": target_Q.mean().item(),
        }

    def update_actor(self, obs: Tensor, step: int) -> Dict:
        """Update the actor network.

        Args:
            obs: Observations.
            step: Global training step.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        std = utils.schedule(self._stddev_schedule, step)
        dist = self._actor(obs, std)
        action = dist.sample(clip=self._stddev_clip)

        Q1, Q2 = self._critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self._actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._actor_opt.step()

        return {"actor_loss": actor_loss.item()}
