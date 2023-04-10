import numpy as np
import torch
from torch.nn import functional as F

from hsuanwu.common.typing import Device, Dict, Storage, Tensor, Tuple
from hsuanwu.xploit.learner import utils
from hsuanwu.xploit.learner.base import BaseLearner
from hsuanwu.xploit.learner.network import DoubleCritic, StochasticActor

DEFAULT_CFGS = {
    'use_aug': False,
    'use_irs': False,
    'num_init_steps': 5000, # only for off-policy algorithms
    # xploit part
    "encoder": {"name": "IdentityEncoder", "observation_space": dict(), "feature_dim": 5},
    "learner": {
        "name": "SACLearner",
        "observation_space": dict(),
        "action_space": dict(),
        "device": str,
        "feature_dim": int,
        "lr": 1e-4,
        "eps": 0.00008,
        "hidden_dim": 1024,
        "critic_target_tau": 0.005,
        "update_every_steps": 2,
        "log_std_range": (-5.0, 2),
        "betas": (0.9, 0.999),
        "temperature":  0.1,
        "fixed_temperature": False,
        "discount": 0.99,
    },
    "storage": {"name": "VanillaReplayStorage", "storage_size": 1000000, "batch_size": 1024},
    # xplore part
    "distribution": {"name": "SquashedNormal"},
    "augmentation": {"name": None},
    "reward": {"name": None},
}

class SACLearner(BaseLearner):
    """Soft Actor-Critic (SAC) Learner

    Args:
        observation_space (Dict): Observation space of the environment. 
            For supporting Hydra, the original 'observation_space' is transformed into a dict like {"shape": observation_space.shape, }.
        action_space (Dict): Action shape of the environment.
            For supporting Hydra, the original 'action_space' is transformed into a dict like 
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or 
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        hidden_dim (int): The size of the hidden layers.
        critic_target_tau (float): The critic Q-function soft-update rate.
        update_every_steps (int): The agent update frequency.
        log_std_range (Tuple[float]): Range of std for sampling actions.
        betas (Tuple[float]): coefficients used for computing running averages of gradient and its square.
        temperature (float): Initial temperature coefficient.
        fixed_temperature (bool): Fixed temperature or not.
        discount (float): Discount factor.

    Returns:
        Soft Actor-Critic learner instance.
    """

    def __init__(
        self,
        observation_space: Dict,
        action_space: Dict,
        device: Device,
        feature_dim: int,
        lr: float,
        eps: float,
        hidden_dim: int,
        critic_target_tau: float,
        update_every_steps: int,
        log_std_range: Tuple[float],
        betas: Tuple[float],
        temperature: float,
        fixed_temperature: bool,
        discount: float,
    ) -> None:
        super().__init__(
            observation_space, action_space, device, feature_dim, lr, eps
        )

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.fixed_temperature = fixed_temperature
        self.discount = discount

        # create models
        self.actor = StochasticActor(
            action_space=action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            log_std_range=log_std_range,
        ).to(self.device)
        self.critic = DoubleCritic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.critic_target = DoubleCritic(
            action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # target entropy
        self.target_entropy = -np.prod(action_space.shape)
        self.log_alpha = torch.tensor(
            np.log(temperature), device=self.device, requires_grad=True
        )

        # create optimizers
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, betas=betas
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr, betas=betas
        )
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr, betas=betas)

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

    @property
    def alpha(self) -> Tensor:
        """Get the temperature coefficient."""
        return self.log_alpha.exp()

    def update(self, replay_storage: Storage, step: int = 0) -> Dict[str, float]:
        """Update the learner.

        Args:
            replay_storage (Storage): Hsuanwu replay storage.
            step (int): Global training step.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """
        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, terminated, next_obs = replay_storage.sample()

        if self.irs is not None:
            intrinsic_reward = self.irs.compute_irs(
                rollouts={
                    "observations": obs.unsqueeze(1).numpy(),
                    "actions": action.unsqueeze(1).numpy(),
                },
                step=step,
            )
            reward += torch.as_tensor(intrinsic_reward, dtype=torch.float32).squeeze(1)

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
                encoded_obs, action, reward, terminated, encoded_next_obs, step
            )
        )

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor_and_alpha(encoded_obs.detach(), step))

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
        terminated: Tensor,
        next_obs: Tensor,
        step: int,
    ) -> Dict[str, float]:
        """Update the critic network.

        Args:
            obs (Tensor): Observations.
            action (Tensor): Actions.
            reward (Tensor): Rewards.
            terminated (Tensor): Terminateds.
            next_obs (Tensor): Next observations.
            step (int): Global training step.

        Returns:
            Critic loss metrics.
        """
        with torch.no_grad():
            dist = self.actor.get_action(next_obs, step=step)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (1.0 - terminated) * self.discount * target_V

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

    def update_actor_and_alpha(self, obs: Tensor, step: int) -> Dict[str, float]:
        """Update the actor network and temperature.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self.actor.get_action(obs, step=step)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if not self.fixed_temperature:
            # update temperature
            self.log_alpha_opt.zero_grad(set_to_none=True)
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            alpha_loss.backward()
            self.log_alpha_opt.step()
        else:
            alpha_loss = torch.scalar_tensor(s=0.0)

        return {"actor_loss": actor_loss.item(), "alpha_loss": alpha_loss.item()}
