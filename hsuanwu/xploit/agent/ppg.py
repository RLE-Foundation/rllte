from typing import Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from omegaconf import DictConfig
import os
from pathlib import Path
from torch import nn

from hsuanwu.xploit.agent.base import BaseAgent
from hsuanwu.xploit.agent.network import ActorCritic
from hsuanwu.xploit.storage import VanillaRolloutStorage as Storage

MATCH_KEYS = {
    "trainer": "OnPolicyTrainer",
    "storage": ["VanillaRolloutStorage"],
    "distribution": ["Categorical", "DiagonalGaussian"],
    "augmentation": [],
    "reward": [],
}

DEFAULT_CFGS = {
    ## TODO: Train setup
    "device": "cpu",
    "seed": 1,
    "num_train_steps": 25000000,
    "num_steps": 256,  # The sample length of per rollout.
    ## TODO: Test setup
    "test_every_episodes": 10,  # only for on-policy algorithms
    "num_test_episodes": 10,
    ## TODO: xploit part
    "encoder": {
        "name": "EspeholtResidualEncoder",
        "observation_space": dict(),
        "feature_dim": 256,
    },
    "agent": {
        "name": "PPG",
        "observation_space": dict(),
        "action_space": dict(),
        "device": str,
        "feature_dim": int,
        "lr": 5e-4,
        "eps": 1e-5,
        "hidden_dim": 256,
        "clip_range": 0.2,
        "num_policy_mini_batch": 8,
        "num_aux_mini_batch": 4,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "policy_epochs": 32,
        "aux_epochs": 6,
        "kl_coef": 1.0,
        "num_aux_grad_accum": 1,
    },
    "storage": {"name": "VanillaRolloutStorage"},
    ## TODO: xplore part
    "distribution": {"name": "Categorical"},
    "augmentation": {"name": None},
    "reward": {"name": None},
}


class PPG(BaseAgent):
    """Phasic Policy Gradient (PPG) agent.
        Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py

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
        clip_range (float): Clipping parameter.
        num_policy_mini_batch (int): Number of mini-batches in policy phase.
        num_aux_mini_batch (int) Number of mini-batches in auxiliary phase.
        vf_coef (float): Weighting coefficient of value loss.
        ent_coef (float): Weighting coefficient of entropy bonus.
        aug_coef (float): Weighting coefficient of augmentation loss.
        max_grad_norm (float): Maximum norm of gradients.
        policy_epochs (int): Number of iterations in the policy phase.
        aux_epochs (int): Number of iterations in the auxiliary phase.
        kl_coef (float): Weighting coefficient of divergence loss.
        num_aux_grad_accum (int): Number of gradient accumulation for auxiliary phase update.

    Returns:
        PPG agent instance.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str,
        feature_dim: int = 256,
        lr: float = 5e-4,
        eps: float = 1e-5,
        hidden_dim: int = 256,
        clip_range: float = 0.2,
        num_policy_mini_batch: int = 8,
        num_aux_mini_batch: int = 4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        aug_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        policy_epochs: int = 32,
        aux_epochs: int = 6,
        kl_coef: float = 1.0,
        num_aux_grad_accum: int = 1,
    ) -> None:
        super().__init__(observation_space, action_space, device, feature_dim, lr, eps)

        self.clip_range = clip_range
        self.num_policy_mini_batch = num_policy_mini_batch
        self.num_aux_mini_batch = num_aux_mini_batch
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.aug_coef = aug_coef
        self.max_grad_norm = max_grad_norm
        self.policy_epochs = policy_epochs
        self.aux_epochs = aux_epochs
        self.kl_coef = kl_coef
        self.num_aux_grad_accum = num_aux_grad_accum

        # auxiliary storage
        self.aux_obs = None
        self.aux_returns = None
        # self.aux_logits = None
        self.aux_policy_outputs = None

        # create models
        self.encoder = None
        # create models
        self.ac = ActorCritic(
            action_shape=self.action_shape,
            action_type=self.action_type,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            aux_critic=True,
        ).to(self.device)

        self.ac_opt = th.optim.Adam(self.ac.parameters(), lr=lr, eps=eps)
        self.train()

    def train(self, training: bool = True) -> None:
        """Set the train mode.

        Args:
            training (bool): True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training
        self.ac.train(training)
        if self.encoder is not None:
            self.encoder.train(training)

    def integrate(self, **kwargs) -> None:
        """Integrate agent and other modules (encoder, reward, ...) together
        """
        self.encoder = kwargs['encoder']
        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=self.lr, eps=self.eps)
        self.encoder.train()
        self.dist = kwargs['dist']
        self.ac.dist = kwargs['dist']
        if kwargs['aug'] is not None:
            self.aug = kwargs['aug']
        if kwargs['irs'] is not None:
            self.irs = kwargs['irs']

    def get_value(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        encoded_obs = self.encoder(obs)
        return self.ac.get_value(obs=encoded_obs)

    def act(self, obs: th.Tensor, training: bool = True, step: int = 0) -> Tuple[th.Tensor, ...]:
        """Sample actions based on observations.

        Args:
            obs: Observations.
            training: training mode, True or False.
            step: Global training step.

        Returns:
            Sampled actions.
        """
        encoded_obs = self.encoder(obs)
        if training:
            actions, values, log_probs, entropy = self.ac.get_action_and_value(obs=encoded_obs)
            return actions.clamp(*self.action_range), values, log_probs, entropy
        else:
            actions = self.ac.get_det_action(obs=encoded_obs)
            return actions.clamp(*self.action_range)

    def update(self, rollout_storage: Storage, episode: int = 0) -> Dict[str, float]:
        """Update the agent.

        Args:
            rollout_storage (Storage): Hsuanwu rollout storage.
            episode (int): Global training episode.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """

        # TODO: Save auxiliary transitions
        if episode == 0:
            num_steps, num_envs = rollout_storage.actions.size()[:2]
            self.aux_obs = th.empty(
                size=(
                    num_steps,
                    num_envs * self.policy_epochs,
                    *self.obs_shape,
                ),
                device="cpu",
                dtype=th.float32,
            )
            self.aux_returns = th.empty(
                size=(num_steps, num_envs * self.policy_epochs),
                device="cpu",
                dtype=th.float32,
            )
            # self.aux_logits = th.empty(
            #     size=(
            #         num_steps,
            #         num_envs * self.policy_epochs,
            #         self.action_shape[0],
            #     ),
            #     device="cpu",
            #     dtype=th.float32,
            # )
            if self.action_type == "Discrete":
                self.aux_policy_outputs = th.empty(
                    size=(
                        num_steps,
                        num_envs * self.policy_epochs,
                        self.action_shape[0],
                    ),
                    device="cpu",
                    dtype=th.float32,
                )
            elif self.action_type == "Box":
                self.aux_policy_outputs = th.empty(
                    size=(
                        num_steps,
                        num_envs * self.policy_epochs,
                        self.action_shape[0] * 2,
                    ),
                    device="cpu",
                    dtype=th.float32,
                )
            else:
                raise NotImplementedError
            self.num_aux_rollouts = num_envs * self.policy_epochs
            self.num_envs = num_envs
            self.num_steps = num_steps

        idx = int(episode % self.policy_epochs)
        self.aux_obs[:, idx * self.num_envs : (idx + 1) * self.num_envs].copy_(rollout_storage.obs[:-1].clone())
        self.aux_returns[:, idx * self.num_envs : (idx + 1) * self.num_envs].copy_(rollout_storage.returns.clone())

        # TODO: Policy phase
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        generator = rollout_storage.sample(self.num_policy_mini_batch)

        if self.irs is not None:
            intrinsic_reward = self.irs.compute_irs(
                samples={
                    "obs": rollout_storage.obs[:-1],
                    "actions": rollout_storage.actions,
                    "next_obs": rollout_storage.obs[1:],
                },
                step=episode * self.num_envs * self.num_steps,
            )
            rollout_storage.rewards += intrinsic_reward.to(self.device)

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

            # evaluate sampled actions
            _, values, log_probs, entropy = self.ac.get_action_and_value(obs=self.encoder(batch_obs), actions=batch_actions)

            # actor loss part
            ratio = th.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * adv_targ
            surr2 = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_targ
            actor_loss = -th.min(surr1, surr2).mean()

            # critic loss part
            values_clipped = batch_values + (values - batch_values).clamp(-self.clip_range, self.clip_range)
            values_losses = (batch_values - batch_returns).pow(2)
            values_losses_clipped = (values_clipped - batch_returns).pow(2)
            critic_loss = 0.5 * th.max(values_losses, values_losses_clipped).mean()

            if self.aug is not None:
                # augmentation loss part
                batch_obs_aug = self.aug(batch_obs)
                new_batch_actions, _, _, _ = self.ac.get_action_and_value(obs=self.encoder(batch_obs))

                _, values_aug, log_probs_aug, _ = self.ac.get_action_and_value(
                    obs=self.encoder(batch_obs_aug), actions=new_batch_actions
                )
                action_loss_aug = -log_probs_aug.mean()
                value_loss_aug = 0.5 * (th.detach(values) - values_aug).pow(2).mean()
                aug_loss = self.aug_coef * (action_loss_aug + value_loss_aug)
            else:
                aug_loss = th.scalar_tensor(s=0.0, requires_grad=False, device=critic_loss.device)

            # update
            self.encoder_opt.zero_grad(set_to_none=True)
            self.ac_opt.zero_grad(set_to_none=True)
            (critic_loss * self.vf_coef + actor_loss - entropy * self.ent_coef + aug_loss).backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.ac_opt.step()
            self.encoder_opt.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += entropy.item()
            num_updates += 1

        total_actor_loss /= num_updates
        total_critic_loss /= num_updates
        total_entropy_loss /= num_updates

        if (episode + 1) % self.policy_epochs != 0:
            # if not auxiliary phase, return train loss directly.
            return {
                "actor_loss": total_actor_loss,
                "critic_loss": total_critic_loss,
                "entropy_loss": total_entropy_loss,
            }

        # TODO: Auxiliary phase
        for idx in range(self.policy_epochs):
            with th.no_grad():
                aux_obs = (
                    self.aux_obs[:, idx * self.num_envs : (idx + 1) * self.num_envs]
                    .to(self.device)
                    .reshape(-1, *self.aux_obs.size()[2:])
                )
                # get policy outputs
                policy_outputs = self.ac.get_policy_outputs(self.encoder(aux_obs)).cpu().clone()
                self.aux_policy_outputs[:, idx * self.num_envs : (idx + 1) * self.num_envs] = policy_outputs.reshape(
                    self.num_steps, self.num_envs, self.aux_policy_outputs.size()[2]
                )
                # # get logits
                # logits = self.ac.get_logits(self.encoder(aux_obs)).cpu().clone()
                # self.aux_logits[:, idx * self.num_envs : (idx + 1) * self.num_envs] = logits.reshape(
                #     self.num_steps, self.num_envs, self.aux_logits.size()[2]
                # )

        total_aux_value_loss = 0.0
        total_kl_loss = 0.0

        for e in range(self.aux_epochs):
            print("Auxiliary Phase", e)
            aux_inds = np.arange(self.num_aux_rollouts)
            np.random.shuffle(aux_inds)

            for idx in range(0, self.num_aux_rollouts, self.num_aux_mini_batch):
                batch_inds = aux_inds[idx : idx + self.num_aux_mini_batch]
                batch_aux_obs = self.aux_obs[:, batch_inds].reshape(-1, *self.aux_obs.size()[2:]).to(self.device)
                batch_aux_returns = self.aux_returns[:, batch_inds].reshape(-1, *self.aux_returns.size()[2:]).to(self.device)
                # batch_aux_logits = self.aux_logits[:, batch_inds].reshape(-1, *self.aux_logits.size()[2:]).to(self.device)
                batch_aux_policy_outputs = self.aux_policy_outputs[:, batch_inds].reshape(
                    -1, *self.aux_policy_outputs.size()[2:]).to(self.device)

                new_dist, new_values, new_aux_values = self.ac.get_probs_and_aux_value(
                    self.encoder(batch_aux_obs))

                new_values = new_values.view(-1)
                new_aux_values = new_aux_values.view(-1)
                if self.action_type == "Discrete":
                    old_dist = self.dist(batch_aux_policy_outputs)
                elif self.action_type == 'Box':
                    old_dist = self.dist(*batch_aux_policy_outputs.chunk(2, dim=1))
                else:
                    raise NotImplementedError
                # divergence loss
                kl_loss = th.distributions.kl_divergence(old_dist, new_dist).mean()
                # value loss
                value_loss = 0.5 * (new_values - batch_aux_returns).mean()
                aux_value_loss = 0.5 * (new_aux_values - batch_aux_returns).mean()
                # total loss
                (value_loss + aux_value_loss + self.kl_coef * kl_loss).backward()

                if (idx + 1) % self.num_aux_grad_accum == 0:
                    self.encoder_opt.zero_grad(set_to_none=True)
                    self.ac_opt.zero_grad(set_to_none=True)
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.encoder_opt.step()
                    self.ac_opt.step()

                total_aux_value_loss += value_loss.item()
                total_aux_value_loss += aux_value_loss.item()
                total_kl_loss += kl_loss.item()

        return {"aux_value_loss": total_aux_value_loss / self.aux_epochs, "kl_loss": total_kl_loss / self.aux_epochs}

    def save(self, path: Path) -> None:
        """Save models.

        Args:
            path (Path): Storage path.

        Returns:
            None.
        """
        if "pretrained" in str(path): # pretraining
            th.save(self.encoder.state_dict(), path / "encoder.pth")
            th.save(self.ac.state_dict(), path / "actor_critic.pth")
        else:
            th.save(self.encoder, path / "encoder.pth")
            del self.ac.critic
            th.save(self.ac, path / "actor.pth")

    def load(self, path: str) -> None:
        """Load initial parameters.

        Args:
            path (str): Import path.

        Returns:
            None.
        """
        encoder_params = th.load(os.path.join(path, 'encoder.pth'), map_location=self.device)
        actor_critic_params = th.load(os.path.join(path, 'actor_critic.pth'), map_location=self.device)
        self.encoder.load_state_dict(encoder_params)
        self.ac.load_state_dict(actor_critic_params)