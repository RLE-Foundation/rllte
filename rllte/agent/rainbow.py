# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import Optional

import torch as th
from torch.nn import functional as F

from rllte.agent import utils
from rllte.common.prototype import OffPolicyAgent
from rllte.common.type_alias import VecEnv
from rllte.xploit.encoder import IdentityEncoder, MnihCnnEncoder, EspeholtResidualEncoder
from rllte.xploit.policy import OffPolicyDoubleDistributionalQNetwork
from rllte.xploit.storage import NStepReplayStorage


class Rainbow(OffPolicyAgent):
    """Rainbow agent.

    Args:
        env (VecEnv): Vectorized environments for training.
        eval_env (VecEnv): Vectorized environments for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_init_steps (int): Number of initial exploration steps.
        storage_size (int): The capacity of the storage.
        feature_dim (int): Number of features extracted by the encoder.
        batch_size (int): Number of samples per batch to load.
        lr (float): The learning rate.
        eps (float): Term added to the denominator to improve numerical stability.
        hidden_dim (int): The size of the hidden layers.
        tau: The Q-function soft-update rate.
        update_every_steps (int): The update frequency of the policy.
        target_update_freq (int): The frequency of target Q-network update.
        discount (float): Discount factor.
        init_fn (str): Parameters initialization method.

    Returns:
        Rainbow agent instance.
    """

    def __init__(
        self,
        env: VecEnv,
        eval_env: Optional[VecEnv] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
        storage_size: int = 10000,
        feature_dim: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        eps: float = 1e-8,
        hidden_dim: int = 1024,
        tau: float = 1.0,
        update_every_steps: int = 4,
        target_update_freq: int = 1000,
        discount: float = 0.99,
        init_fn: str = "orthogonal",
        encoder_model: str = "mnih",
        v_min: float = -100.0,
        v_max: float = 100.0,
        n_atoms: int = 101,
    ) -> None:
        super().__init__(
            env=env,
            eval_env=eval_env,
            tag=tag,
            seed=seed,
            device=device,
            pretraining=pretraining,
            num_init_steps=num_init_steps,
        )

        # hyper parameters
        self.lr = lr
        self.eps = eps
        self.tau = tau
        self.discount = discount
        self.update_every_steps = update_every_steps
        self.target_update_freq = target_update_freq
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms

        # default encoder
        if len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]  # type: ignore
            encoder = IdentityEncoder(
                observation_space=env.observation_space, feature_dim=feature_dim  # type: ignore[assignment]
            )
        elif encoder_model == "mnih":
            encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif encoder_model == "espeholt":
            encoder = EspeholtResidualEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        else:
            raise NotImplementedError(f"Unsupported encoder model {encoder_model}!")
        
        
        # create policy
        policy = OffPolicyDoubleDistributionalQNetwork(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps),
            init_fn=init_fn,
        )

        # default storage
        storage = NStepReplayStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            storage_size=storage_size,
            device=device,
            num_envs=self.num_envs,
            batch_size=batch_size,
        )

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage)

    def update(self) -> None:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        if self.global_step % self.update_every_steps != 0:
            return None

        # sample a batch
        batch = self.storage.sample()

        # compute intrinsic rewards
        if self.irs is not None:
            intrinsic_rewards = self.irs.compute_irs(
                samples={
                    "obs": batch.observations,
                    "actions": batch.actions,
                    "next_obs": batch.next_observations,
                },
                step=self.global_step,
            )
            batch = batch._replace(reward=batch.rewards + intrinsic_rewards.to(self.device))

        # compute target Q values
        with th.no_grad():
            next_pmfs = self.policy.get_target_dist(batch.next_observations)
            next_atoms = batch.rewards + self.discount * self.policy.target_atoms * (1.0 - batch.terminateds) * (1.0 - batch.truncateds)
            delta_z = self.policy.target_atoms[1] - self.policy.target_atoms[0]
            tz = next_atoms.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = b.floor().clamp(0, self.n_atoms - 1)
            u = b.ceil().clamp(0, self.n_atoms - 1)
            d_m_l = (u + (l == u).float() - b) * next_pmfs
            d_m_u = (b - l) * next_pmfs
            target_pmfs = th.zeros_like(next_pmfs)
            for i in range(target_pmfs.size(0)):
                target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

        # compute current Q values
        old_pmfs = self.policy.get_online_dist(batch.observations, batch.actions.flatten())
        loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
        q_values = (old_pmfs * self.policy.atoms).sum(1)

        # optimize the qnet
        self.policy.optimizers["opt"].zero_grad(set_to_none=True)
        loss.backward()
        self.policy.optimizers["opt"].step()

        # udpate target qnet
        if self.global_step % self.target_update_freq == 0:
            utils.soft_update_params(self.policy.qnet, self.policy.qnet_target, self.tau)
            self.policy.target_atoms.data.copy_(self.policy.atoms.data)

        # record metrics
        self.logger.record("train/q_loss", loss.item())
        self.logger.record("train/q", q_values.mean().item())