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


from typing import Dict, Optional

import gymnasium as gym
import torch as th
from torch.nn import functional as F

from rllte.agent import utils
from rllte.common.prototype import OffPolicyAgent
from rllte.xploit.encoder import IdentityEncoder, MnihCnnEncoder
from rllte.xploit.policy import OffPolicyDoubleQNetwork
from rllte.xploit.storage import VanillaReplayStorage


class DQN(OffPolicyAgent):
    """Deep Q-Network (DQN) agent.

    Args:
        env (gym.Env): A Gym-like environment for training.
        eval_env (gym.Env): A Gym-like environment for evaluation.
        tag (str): An experiment tag.
        seed (int): Random seed for reproduction.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        pretraining (bool): Turn on the pre-training mode.

        num_init_steps (int): Number of initial exploration steps.
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
        DQN agent instance.
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        tag: str = "default",
        seed: int = 1,
        device: str = "cpu",
        pretraining: bool = False,
        num_init_steps: int = 2000,
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

        # default encoder
        if len(self.obs_shape) == 3:
            encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
        elif len(self.obs_shape) == 1:
            feature_dim = self.obs_shape[0]
            encoder = IdentityEncoder(observation_space=env.observation_space, feature_dim=feature_dim)

        # create policy
        policy = OffPolicyDoubleQNetwork(
            observation_space=env.observation_space,
            action_space=env.action_space,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            opt_class=th.optim.Adam,
            opt_kwargs=dict(lr=lr, eps=eps),
            init_fn=init_fn,
        )

        # default storage
        storage = VanillaReplayStorage(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            num_envs=self.num_envs,
            batch_size=batch_size,
            storage_size=10000,
        )

        # set all the modules [essential operation!!!]
        self.set(encoder=encoder, policy=policy, storage=storage)

    def update(self) -> Dict[str, float]:
        """Update the agent and return training metrics such as actor loss, critic_loss, etc."""
        metrics = {}
        if self.global_step % self.update_every_steps != 0:
            return metrics

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

        # encode
        encoded_obs = self.policy.encoder(batch.observations)
        with th.no_grad():
            encoded_next_obs = self.policy.encoder(batch.next_observations)

        # compute target Q values
        with th.no_grad():
            next_q_values = self.policy.qnet_target(encoded_next_obs)
            next_q_values, _ = next_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)
            # time limit mask
            target_q_values = (
                batch.rewards + (1.0 - batch.terminateds) * (1.0 - batch.truncateds) * self.discount * next_q_values
            )

        # compute current Q values
        q_values = self.policy.qnet(encoded_obs)
        q_values = th.gather(q_values, dim=1, index=batch.actions.unsqueeze(1).long())
        # following https://github.com/DLR-RM/stable-baselines3/blob/d68ff2e17f2f823e6f48d9eb9cee28ca563a2554/stable_baselines3/dqn/dqn.py
        # less sensitive to outliers
        huber_loss = F.mse_loss(q_values, target_q_values)

        # optimize the qnet
        self.policy.optimizers['opt'].zero_grad(set_to_none=True)
        huber_loss.backward()
        self.policy.optimizers['opt'].step()

        # udpate target qnet
        if self.global_step % self.target_update_freq:
            utils.soft_update_params(self.policy.qnet, self.policy.qnet_target, self.tau)

        return {"Huber Loss": huber_loss.item(), "Q": q_values.mean().item(), "Target Q": target_q_values.mean().item()}
