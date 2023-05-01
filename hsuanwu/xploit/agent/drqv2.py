from typing import Dict, Generator, Tuple, Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig
import os
from pathlib import Path
from torch.nn import functional as F

from hsuanwu.xploit.agent import utils
from hsuanwu.xploit.agent.base import BaseAgent
from hsuanwu.xploit.agent.network import DeterministicActor, DoubleCritic

MATCH_KEYS = {
    "trainer": "OffPolicyTrainer",
    "storage": ["NStepReplayStorage"],
    "distribution": [
        "OrnsteinUhlenbeckNoise",
        "TruncatedNormalNoise",
        "NormalNoise",
    ],
    "augmentation": [],
    "reward": [],
}

DEFAULT_CFGS = {
    ## TODO: Train setup
    "device": "cpu",
    "seed": 1,
    "num_train_steps": 500000,
    "num_init_steps": 2000,  # only for off-policy algorithms
    ## TODO: Test setup
    "test_every_steps": 5000,  # only for off-policy algorithms
    "num_test_episodes": 10,
    ## TODO: xploit part
    "encoder": {
        "name": "TassaCnnEncoder",
        "observation_space": dict(),
        "feature_dim": 50,
    },
    "agent": {
        "name": "DrQv2",
        "observation_space": dict(),
        "action_space": dict(),
        "device": str,
        "feature_dim": int,
        "lr": 1e-4,
        "eps": 0.00008,
        "hidden_dim": 1024,
        "critic_target_tau": 0.01,
        "update_every_steps": 2,
    },
    "storage": {
        "name": "NStepReplayStorage",
        "storage_size": 500000,
        "batch_size": 256,
    },
    ## TODO: xplore part
    "distribution": {"name": "TruncatedNormalNoise"},
    "augmentation": {"name": "RandomShift"},
    "reward": {"name": None},
}


class DrQv2(BaseAgent):
    """Data Regularized-Q v2 (DrQ-v2).
        When 'augmentation' module is deprecated, this agent will transform into
            Deep Deterministic Policy Gradient (DDPG) agent.
        Based on: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted by the encoder.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        hidden_dim (int): The size of the hidden layers.
        critic_target_tau: The critic Q-function soft-update rate.
        update_every_steps (int): The agent update frequency.

    Returns:
        DrQv2 agent instance.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str,
        feature_dim: int,
        lr: float,
        eps: float,
        hidden_dim: int,
        critic_target_tau: float,
        update_every_steps: int,
    ) -> None:
        super().__init__(observation_space, action_space, device, feature_dim, lr, eps)

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps

        # create models
        self.actor = DeterministicActor(action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self.critic = DoubleCritic(action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = DoubleCritic(action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers
        self.actor_opt = th.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = th.optim.Adam(self.critic.parameters(), lr=self.lr)
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
    
    def integrate(self, **kwargs) -> None:
        """Integrate agent and other modules (encoder, reward, ...) together
        """
        self.encoder = kwargs['encoder']
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=self.lr, eps=self.eps)
        self.encoder.train()
        self.dist = kwargs['dist']
        self.actor.dist = kwargs['dist']
        if kwargs['aug'] is not None:
            self.aug = kwargs['aug']
        if kwargs['irs'] is not None:
            self.irs = kwargs['irs']

    def act(self, obs: th.Tensor, training: bool = True, step: int = 0) -> Tuple[th.Tensor]:
        """Sample actions based on observations.

        Args:
            obs (Tensor): Observations.
            training (bool): training mode, True or False.
            step (int): Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        dist = self.actor.get_dist(obs=encoded_obs, step=step)

        if not training:
            action = dist.mean
        else:
            action = dist.sample()

        return action.clamp(*self.action_range)

    def update(self, replay_iter: Generator, step: int = 0) -> Dict[str, float]:
        """Update the agent.

        Args:
            replay_iter (Generator): Hsuanwu replay storage iterable dataloader.
            step (int): Global training step.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """

        metrics = {}
        if step % self.update_every_steps != 0:
            return metrics

        obs, action, reward, discount, next_obs = next(replay_iter)
        obs = obs.float().to(self.device)
        action = action.float().to(self.device)
        reward = reward.float().to(self.device)
        discount = discount.float().to(self.device)
        next_obs = next_obs.float().to(self.device)

        if self.irs is not None:
            intrinsic_reward = self.irs.compute_irs(
                samples={
                    "obs": obs.unsqueeze(1),
                    "actions": action.unsqueeze(1),
                    "next_obs": next_obs.unsqueeze(1),
                },
                step=step,
            )
            reward += intrinsic_reward.to(self.device)

        # obs augmentation
        if self.aug is not None:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())

        # encode
        encoded_obs = self.encoder(obs)
        with th.no_grad():
            encoded_next_obs = self.encoder(next_obs)

        # update criitc
        metrics.update(self.update_critic(encoded_obs, action, reward, discount, encoded_next_obs, step))

        # update actor (do not udpate encoder)
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # udpate critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def update_critic(
        self,
        obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        discount: th.Tensor,
        next_obs: th.Tensor,
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

        with th.no_grad():
            # sample actions
            dist = self.actor.get_dist(next_obs, step=step)

            next_action = dist.sample(clip=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = th.min(target_Q1, target_Q2)
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

    def update_actor(self, obs: th.Tensor, step: int) -> Dict[str, float]:
        """Update the actor network.

        Args:
            obs (Tensor): Observations.
            step (int): Global training step.

        Returns:
            Actor loss metrics.
        """
        # sample actions
        dist = self.actor.get_dist(obs, step=step)
        action = dist.sample(clip=True)

        Q1, Q2 = self.critic(obs, action)
        Q = th.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return {"actor_loss": actor_loss.item()}

    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Storage path.

        Returns:
            None.
        """
        if "pretrained" in str(path): # pretraining
            th.save(self.encoder.state_dict(), path / "encoder.pth")
            th.save(self.actor.state_dict(), path / "actor.pth")
            th.save(self.critic.state_dict(), path / "critic.pth")
        else:
            th.save(self.encoder, path / "encoder.pth")
            th.save(self.actor, path / "actor.pth")
            th.save(self.critic, path / "critic.pth")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        encoder_params = th.load(os.path.join(path, 'encoder.pth'), map_location=self.device)
        actor_params = th.load(os.path.join(path, 'actor.pth'), map_location=self.device)
        critic_params = th.load(os.path.join(path, 'critic.pth'), map_location=self.device)
        self.encoder.load_state_dict(encoder_params)
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(critic_params)