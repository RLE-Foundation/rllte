import numpy as np
import torch
from torch import nn

from hsuanwu.common.typing import Space, Tensor, Dict, Device, Storage
from hsuanwu.xploit.learner.base import BaseLearner
from hsuanwu.xploit.learner.network import DiscreteActorAuxiliaryCritic

class PPGLearner(BaseLearner):
    """Phasic Policy Gradient (PPG) Learner.

    Args:
        observation_space (Space): Observation space of the environment.
        action_space (Space): Action shape of the environment.
        action_type (str): Continuous or discrete action. "cont" or "dis".
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        feature_dim (int): Number of features extracted.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.

        hidden_dim (int): The size of the hidden layers.
        clip_range (float): Clipping parameter.
        num_policy_mini_batch (int): Number of mini-batches in policy phase.
        num_aux_mini_batch (int) Number of mini-batches in auxiliary phase.
        vf_coef (float): Weighting coefficient of value loss.
        ent_coef (float): Weighting coefficient of entropy bonus.
        max_grad_norm (float): Maximum norm of gradients.
        policy_epochs (int): Number of iterations in the policy phase.
        aux_epochs (int): Number of iterations in the auxiliary phase.
        kl_coef (float): Weighting coefficient of divergence loss.
        num_aux_grad_accum (int): Number of gradient accumulation for auxiliary phase update.

    Returns:
        PPG learner instance.
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        action_type: str,
        device: Device,
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
        super().__init__(
            observation_space, action_space, action_type, device, feature_dim, lr, eps
        )

        self.clip_range = clip_range
        self.num_policy_mini_batch = num_policy_mini_batch
        self.num_aux_mini_batch = num_aux_mini_batch
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.policy_epochs = policy_epochs
        self.aux_epochs = aux_epochs
        self.kl_coef = kl_coef
        self.num_aux_grad_accum = num_aux_grad_accum

        # auxiliary storage
        self.aux_obs = None
        self.aux_returns = None
        self.aux_logits = None

        # create models
        self.encoder = None
        # create models
        if self.action_type == 'dis':
            self.ac = DiscreteActorAuxiliaryCritic(
                action_space=action_space, feature_dim=feature_dim, hidden_dim=hidden_dim
            ).to(self.device)
        else:
            raise NotImplementedError

        self.ac_opt = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=eps)
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

    def get_value(self, obs: Tensor) -> Tensor:
        """Get estimated values for observations.

        Args:
            obs (Tensor): Observations.

        Returns:
            Estimated values.
        """
        encoded_obs = self.encoder(obs)
        return self.ac.get_value(obs=encoded_obs)

    def update(self, rollout_storage: Storage, episode: int = 0) -> Dict[str, float]:
        """Update the learner.

        Args:
            rollout_storage (Storage): Hsuanwu rollout storage.
            episode (int): Global training episode.

        Returns:
            Training metrics such as actor loss, critic_loss, etc.
        """

        # TODO: Save auxiliary transitions
        if episode == 0:
            num_steps, num_envs = rollout_storage.obs.size()[:2]
            self.aux_obs = torch.empty(
                size=(
                    num_steps,
                    num_envs * self.policy_epochs,
                    *self.obs_space.shape,
                ),
                device="cpu",
                dtype=torch.float32,
            )
            self.aux_returns = torch.empty(
                size=(num_steps, num_envs * self.policy_epochs, 1),
                device="cpu",
                dtype=torch.float32,
            )
            self.aux_logits = torch.empty(
                size=(
                    num_steps,
                    num_envs * self.policy_epochs,
                    self.action_space.shape[0],
                ),
                device="cpu",
                dtype=torch.float32,
            )
            self.num_aux_rollouts = num_envs * self.policy_epochs
            self.num_envs = num_envs
            self.num_steps = num_steps

        idx = int(episode % self.policy_epochs)
        self.aux_obs[:, idx * self.num_envs : (idx + 1) * self.num_envs].copy_(
            rollout_storage.obs.clone()
        )
        self.aux_returns[:, idx * self.num_envs : (idx + 1) * self.num_envs].copy_(
            rollout_storage.returns.clone()
        )

        # TODO: Policy phase
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        num_updates = 0

        generator = rollout_storage.generator(self.num_policy_mini_batch)

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
            _, values, log_probs, entropy = self.ac.get_action_and_value(
                obs=self.encoder(batch_obs), actions=batch_actions
            )

            # actor loss part
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * adv_targ
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                * adv_targ
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # critic loss part
            values_clipped = batch_values + (values - batch_values).clamp(
                -self.clip_range, self.clip_range
            )
            values_losses = (batch_values - batch_returns).pow(2)
            values_losses_clipped = (values_clipped - batch_returns).pow(2)
            critic_loss = 0.5 * torch.max(values_losses, values_losses_clipped).mean()

            # update
            self.encoder_opt.zero_grad(set_to_none=True)
            self.ac_opt.zero_grad(set_to_none=True)
            (
                critic_loss * self.vf_coef + actor_loss - entropy * self.ent_coef
            ).backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.encoder_opt.step()
            self.ac_opt.step()

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
            with torch.no_grad():
                aux_obs = (
                    self.aux_obs[:, idx * self.num_envs : (idx + 1) * self.num_envs]
                    .to(self.device)
                    .reshape(-1, *self.aux_obs.size()[2:])
                )
                # get logits
                logits = (
                    self.ac.get_logits(self.encoder(aux_obs)).logits.cpu().clone()
                )
                self.aux_logits[
                    :, idx * self.num_envs : (idx + 1) * self.num_envs
                ] = logits.reshape(
                    self.num_steps, self.num_envs, self.aux_logits.size()[2]
                )

        for e in range(self.aux_epochs):
            print("Auxiliary Phase", e)
            aux_inds = np.arange(self.num_aux_rollouts)
            np.random.shuffle(aux_inds)

            for idx in range(0, self.num_aux_rollouts, self.num_aux_mini_batch):
                batch_inds = aux_inds[idx : idx + self.num_aux_mini_batch]
                batch_aux_obs = (
                    self.aux_obs[:, batch_inds]
                    .reshape(-1, *self.aux_obs.size()[2:])
                    .to(self.device)
                )
                batch_aux_returns = (
                    self.aux_returns[:, batch_inds]
                    .reshape(-1, *self.aux_returns.size()[2:])
                    .to(self.device)
                )
                batch_aux_logits = (
                    self.aux_logits[:, batch_inds]
                    .reshape(-1, *self.aux_logits.size()[2:])
                    .to(self.device)
                )

                new_dist, new_values, new_aux_values = self.ac.get_probs_and_aux_value(
                    self.encoder(batch_aux_obs)
                )

                new_values = new_values.view(-1)
                new_aux_values = new_aux_values.view(-1)
                old_dist = self.dist(logits=batch_aux_logits)
                # divergence loss
                kl_loss = torch.distributions.kl_divergence(old_dist, new_dist).mean()
                # value loss
                value_loss = 0.5 * ((new_values - batch_aux_returns)).mean()
                aux_value_loss = 0.5 * ((new_aux_values - batch_aux_returns)).mean()
                # total loss
                (value_loss + aux_value_loss + self.kl_coef * kl_loss).backward()

                if (idx + 1) % self.num_aux_grad_accum == 0:
                    self.encoder_opt.zero_grad(set_to_none=True)
                    self.ac_opt.zero_grad(set_to_none=True)
                    nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), self.max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.encoder_opt.step()
                    self.ac_opt.step()
