import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union, Optional

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from rllte.common.on_policy_agent import OnPolicyAgent
from rllte.common.utils import get_network_init


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
                         eval_every_episodes=eval_every_episodes,
                         shared_encoder=True,
                         feature_dim=feature_dim,
                         hidden_dim=hidden_dim,
                         batch_size=batch_size,
                         npu=True)
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
    
    def freeze(self) -> None:
        """Freeze the structure of the agent."""
        # set encoder and distribution
        self.policy.encoder = self.encoder
        self.policy.dist = self.dist
        # network initialization
        self.policy.apply(get_network_init(self.network_init_method))
        # to device
        self.policy.to(self.device)
        # create optimizers
        self.opt = th.optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        # set the training mode
        self.mode(training=True)

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
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch_obs, actions=batch_actions)

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
                    new_batch_actions, _, _ = self.policy.get_action_and_value(obs=batch_obs)

                    values_aug, log_probs_aug, _ = self.policy.evaluate_actions(obs=batch_obs_aug, actions=new_batch_actions)
                    action_loss_aug = -log_probs_aug.mean()
                    value_loss_aug = 0.5 * (th.detach(new_values) - values_aug).pow(2).mean()
                    aug_loss = self.aug_coef * (action_loss_aug + value_loss_aug)
                else:
                    aug_loss = th.scalar_tensor(s=0.0, requires_grad=False, device=critic_loss.device)

                # update
                self.opt.zero_grad(set_to_none=True)
                loss = critic_loss * self.vf_coef + actor_loss - entropy * self.ent_coef + aug_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

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
