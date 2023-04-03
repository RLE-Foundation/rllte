import torch
from torch.nn import functional as F

from hsuanwu.common.typing import Device, Dict, Iterable, Space, Tensor
from hsuanwu.xploit import utils
from hsuanwu.xploit.learner.base import BaseLearner
from hsuanwu.xploit.learner.network import DeterministicActor, DoubleCritic


class DrQv2Learner(BaseLearner):
    """Data Regularized-Q v2 (DrQ-v2).

    Args:
        observation_space (Space): Observation space of the environment.
        action_space (Space): Action shape of the environment.
        action_type (str): Continuous or discrete action. "cont" or "dis".
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        hidden_dim (int): The size of the hidden layers.
        critic_target_tau: The critic Q-function soft-update rate.
        update_every_steps (int): The agent update frequency.

    Returns:
        DrQv2 learner instance.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: Device = "cuda",
        feature_dim: int = 50,
        lr: float = 1e-4,
        eps: float = 0.00008,
        hidden_dim: int = 1024,
        critic_target_tau: float = 0.01,
        update_every_steps: int = 2,
    ) -> None:
        super().__init__(
            observation_space, action_space, action_type, device, feature_dim, lr, eps
        )

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps

        # create models
        self.actor = DeterministicActor(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.critic = DoubleCritic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.critic_target = DoubleCritic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.train()
        self.critic_target.train()

    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder is not None:
            self.encoder.train(training)

    def update(self, replay_iter: Iterable, step: int = 0) -> Dict[str, float]:
        """Update the learner.

        Args:
            replay_iter (Iterable): Hsuanwu replay storage iterable dataloader.
            step (int): Global training step.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """

        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, discount, next_obs = next(replay_iter)
        if self.irs is not None:
            intrinsic_reward = self.irs.compute_irs(
                rollouts={
                    "observations": obs.unsqueeze(1).numpy(),
                    "actions": action.unsqueeze(1).numpy(),
                },
                step=step,
            )
            reward += torch.as_tensor(intrinsic_reward, dtype=torch.float32).squeeze(1)

        obs = obs.float().to(self.device)
        action = action.float().to(self.device)
        reward = reward.float().to(self.device)
        discount = discount.float().to(self.device)
        next_obs = next_obs.float().to(self.device)

        # obs augmentation
        if self.aug is not None:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())

        # encode
        encoded_obs = self.encoder(obs)
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_obs)

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
            self.critic, self.critic_target, self.critic_target_tau
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
    ) -> Dict[str, float]:
        """Update the critic network.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.
            reward (Tensor): Rewards.
            discount (Tensor): discounts.
            next_obs (Tensor): Next observations.
            step (int): Global training step.

        Returns:
            Critic loss metrics.
        """

        with torch.no_grad():
            # sample actions
            dist = self.actor.get_action(next_obs, step=step)

            next_action = dist.sample(clip=True)
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

        return {
            "critic_loss": critic_loss.item(),
            "critic_q1": Q1.mean().item(),
            "critic_q2": Q2.mean().item(),
            "critic_target": target_Q.mean().item(),
        }

    def update_actor(self, obs: Tensor, step: int) -> Dict[str, float]:
        """Update the actor network.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self.actor.get_action(obs, step=step)
        action = dist.sample(clip=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return {"actor_loss": actor_loss.item()}
