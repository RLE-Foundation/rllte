import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from hsuanwu.common.on_policy_agent import OnPolicyAgent
from hsuanwu.xploit.agent.networks import NpuOnPolicySharedActorCritic, get_network_init
from hsuanwu.xploit.encoder import MnihCnnEncoder, IdentityEncoder
from hsuanwu.xploit.storage import VanillaRolloutStorage as Storage
from hsuanwu.xplore.distribution import Categorical, DiagonalGaussian, Bernoulli


class NpuPPO(OnPolicyAgent):
    """Proximal Policy Optimization (PPO) agent for `NPU` device.
        When the `augmentation` module is invoked, this agent will transform into Data Regularized Actor-Critic (DrAC) agent.
        Based on: https://github.com/yuanmingqi/pytorch-a2c-ppo-acktr-gail

    Args:
        env (Env): A Gym-like environment for training.
        eval_env (Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_steps (int): The sample length of per rollout.
        eval_every_episodes (int): Evaluation interval.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.
        hidden_dim (int): The size of the hidden layers.
        clip_range (float): Clipping parameter.
        clip_range_vf (float): Clipping parameter for the value function.
        n_epochs (int): Times of updating the policy.
        vf_coef (float): Weighting coefficient of value loss.
        ent_coef (float): Weighting coefficient of entropy bonus.
        aug_coef (float): Weighting coefficient of augmentation loss.
        max_grad_norm (float): Maximum norm of gradients.
        network_init_method (str): Network initialization method name.

    Returns:
        PPO agent instance.
    """

    def __init__(
        self,
        env: gym.Env, 
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_steps: int = 128,
        eval_every_episodes: int = 10,
        feature_dim: int = 512,
        batch_size: int = 256,
        lr: float = 2.5e-4,
        eps: float = 1e-5,
        hidden_dim: int = 512,
        clip_range: float = 0.1,
        clip_range_vf: float = 0.1,
        n_epochs: int = 4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        aug_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        network_init_method: str = "orthogonal",
    ) -> None:
        super().__init__(env=env,
                         eval_env=eval_env,
                         tag=tag,
                         seed=seed,
                         device=device,
                         pretraining=pretraining,
                         num_steps=num_steps,
                         eval_every_episodes=eval_every_episodes)
        self.feature_dim = feature_dim
        self.lr = lr
        self.eps = eps
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.aug_coef = aug_coef
        self.max_grad_norm = max_grad_norm
        self.network_init_method = network_init_method

        # build encoder
        if len(self.obs_shape) == 3:
            self.encoder = MnihCnnEncoder(
                observation_space=env.observation_space,
                feature_dim=feature_dim
            )
        elif len(self.obs_shape) == 1:
            self.encoder = IdentityEncoder(
                observation_space=env.observation_space,
                feature_dim=feature_dim
            )
            self.feature_dim = self.obs_shape[0]

        # build storage
        self.storage = Storage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            batch_size=batch_size
        )

        # build distribution
        if self.action_type == "Discrete":
            self.dist = Categorical
        elif self.action_type == "Box":
            self.dist = DiagonalGaussian
        elif self.action_type == "MultiBinary":
            self.dist = Bernoulli
        else:
            raise NotImplementedError("Unsupported action type!")

        # create models
        self.ac = NpuOnPolicySharedActorCritic(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            action_type=self.action_type,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
        )

    def mode(self, training: bool = True) -> None:
        """Set the training mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.ac.train(training)

    def set(self, 
            encoder: Optional[Any] = None,
            storage: Optional[Any] = None,
            distribution: Optional[Any] = None,
            augmentation: Optional[Any] = None,
            reward: Optional[Any] = None,
            ) -> None:
        """Set a module for the agent.

        Args:
            encoder (Optional[Any]): An encoder of `hsuanwu.xploit.encoder` or a custom encoder.
            storage (Optional[Any]): A storage of `hsuanwu.xploit.storage` or a custom storage.
            distribution (Optional[Any]): A distribution of `hsuanwu.xplore.distribution` or a custom distribution.
            augmentation (Optional[Any]): An augmentation of `hsuanwu.xplore.augmentation` or a custom augmentation.
            reward (Optional[Any]): A reward of `hsuanwu.xplore.reward` or a custom reward.

        Returns:
            None.
        """
        super().set(
            encoder=encoder,
            storage=storage,
            distribution=distribution,
            augmentation=augmentation,
            reward=reward
        )
        if encoder is not None:
            self.encoder = encoder
            assert self.encoder.feature_dim == self.feature_dim, "The `feature_dim` argument of agent and encoder must be same!"

    def freeze(self) -> None:
        """Freeze the structure of the agent."""
        # set encoder and distribution
        self.ac.encoder = self.encoder
        self.ac.dist = self.dist
        # network initialization
        self.ac.apply(get_network_init(self.network_init_method))
        # to device
        self.ac.to(self.device)
        # create optimizers
        self.ac_opt = th.optim.Adam(self.ac.parameters(), lr=self.lr, eps=self.eps)
        # set the training mode
        self.mode(training=True)

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        return self.ac.get_value(obs).cpu()

    def act(self, obs: th.Tensor, training: bool = True, step: int = 0) -> Union[Tuple[th.Tensor, ...], Dict[str, Any]]:
        """Sample actions based on observations.

        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
        if training:
            actions, values, log_probs = self.ac.get_action_and_value(obs)
            return {"actions": actions, "values": values, "log_probs": log_probs}
        else:
            actions = self.ac.get_det_action(obs)
            return actions

    def update(self) -> Dict[str, float]:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc.
        """
        total_actor_loss = [0.]
        total_critic_loss = [0.]
        total_entropy_loss = [0.]
        total_aug_loss = [0.]

        for _ in range(self.n_epochs):
            generator = self.storage.sample()

            for batch in generator:
                (
                    batch_obs,
                    batch_actions,
                    batch_values,
                    batch_returns,
                    batch_terminateds,
                    batch_truncateds,
                    batch_old_log_probs,
                    adv_targ,
                ) = batch

                # to device
                batch_obs = batch_obs.to(self.device)
                batch_values = batch_values.to(self.device)
                batch_returns = batch_returns.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                adv_targ = adv_targ.to(self.device)

                # evaluate sampled actions
                new_values, new_log_probs, entropy = self.ac.evaluate_actions(obs=batch_obs, actions=batch_actions)

                # to device
                new_values = new_values.to(self.device)
                new_log_probs = new_log_probs.to(self.device)
                entropy = entropy.to(self.device)

                # actor loss part
                ratio = th.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * adv_targ
                surr2 = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_targ
                actor_loss = -th.min(surr1, surr2).mean()

                # critic loss part
                if self.clip_range_vf is None:
                    critic_loss = 0.5 * (new_values.flatten() - batch_returns).pow(2).mean()
                else:
                    values_clipped = batch_values + (new_values.flatten() - batch_values).clamp(
                        -self.clip_range_vf, self.clip_range_vf
                    )
                    values_losses = (new_values.flatten() - batch_returns).pow(2)
                    values_losses_clipped = (values_clipped - batch_returns).pow(2)
                    critic_loss = 0.5 * th.max(values_losses, values_losses_clipped).mean()

                if self.aug is not None:
                    # augmentation loss part
                    batch_obs_aug = self.aug(batch_obs)
                    new_batch_actions, _, _ = self.ac.get_action_and_value(obs=batch_obs)

                    values_aug, log_probs_aug, _ = self.ac.evaluate_actions(obs=batch_obs_aug, actions=new_batch_actions)
                    action_loss_aug = -log_probs_aug.mean()
                    value_loss_aug = 0.5 * (th.detach(new_values) - values_aug).pow(2).mean()
                    aug_loss = self.aug_coef * (action_loss_aug + value_loss_aug)
                else:
                    aug_loss = th.scalar_tensor(s=0.0, requires_grad=False, device=critic_loss.device)

                # update
                self.ac_opt.zero_grad(set_to_none=True)
                loss = critic_loss * self.vf_coef + actor_loss - entropy * self.ent_coef + aug_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.ac_opt.step()

                total_actor_loss.append(actor_loss.item())
                total_critic_loss.append(critic_loss.item())
                total_entropy_loss.append(entropy.item())
                total_aug_loss.append(aug_loss.item())

        return {
            "actor_loss": np.mean(total_actor_loss),
            "critic_loss": np.mean(total_critic_loss),
            "entropy": np.mean(total_entropy_loss),
            "aug_loss": np.mean(total_aug_loss),
        }

    def save(self) -> None:
        """Save models."""
        if self.pretraining:  # pretraining
            save_dir = Path.cwd() / "pretrained"
            save_dir.mkdir(exist_ok=True)
            th.save(self.ac.state_dict(), save_dir / "actor_critic.pth")
        else:
            save_dir = Path.cwd() / "model"
            save_dir.mkdir(exist_ok=True)
            del self.ac.critic, self.ac.dist
            th.save(self.ac, save_dir / "agent.pth")

        self.logger.info(f"Model saved at: {save_dir}")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        self.logger.info(f"Loading Initial Parameters from {path}")
        actor_critic_params = th.load(os.path.join(path, "actor_critic.pth"), map_location=self.device)
        self.ac.load_state_dict(actor_critic_params)
