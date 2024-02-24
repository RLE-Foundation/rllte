# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

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


from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch as th
from tqdm import tqdm
import rllte

from rllte.common.preprocessing import process_action_space, process_observation_space
from rllte.common.utils import TorchRunningMeanStd, RewardForwardFilter

class BaseReward(ABC):
    """Base class of reward module.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        n_envs (int): The number of parallel environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        rwd_norm_type (bool): Use running mean and std for reward normalization, or minmax or none
        obs_rms (bool): Use running mean and std for observation normalization.
        gamma (Optional[float]): Intrinsic reward discount rate, None for no discount.

    Returns:
        Instance of the base reward module.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        rwd_norm_type: str = "rms",
        obs_rms: bool = False,
        gamma: Optional[float] = None
    ) -> None:
        # get environment information
        self.obs_shape: Tuple = process_observation_space(observation_space)  # type: ignore
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type \
            = process_action_space(action_space)
        self.n_envs = n_envs

        # set device and parameters
        self.device = th.device(device)
        self.beta = beta
        self.kappa = kappa
        self.rwd_norm_type = rwd_norm_type
        self.obs_rms = obs_rms
        self.global_step = 0

        # build the running mean and std for normalization
        self.rms = TorchRunningMeanStd() if "rms" in self.rwd_norm_type else None
        self.obs_norm = TorchRunningMeanStd(shape=self.obs_shape) if self.obs_rms else None

        # build the reward forward filter
        self.rff = RewardForwardFilter(gamma) if gamma is not None else None
        
        # build logger
        self.logger = None
        
    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards.
        """
        return self.beta * np.power(1.0 - self.kappa, self.global_step)
    
    def scale(self, rewards: th.Tensor) -> th.Tensor:
        """Scale the intrinsic rewards.

        Args:
            rewards (th.Tensor): The intrinsic rewards with shape (n_steps, n_envs).
        
        Returns:
            The scaled intrinsic rewards.
        """
        if self.rff is not None:
            for step in range(rewards.size(0)):
                rewards[step] = self.rff.update(rewards[step])

        if self.rwd_norm_type == "rms":
            self.rms.update(rewards.ravel())
            std_rewards = ((rewards) / self.rms.std) * self.weight
            return std_rewards
        elif self.rwd_norm_type == "clipped-rms":
            self.rms.update(rewards.ravel())
            std_rewards = th.max((rewards / self.rms.std) - (self.rms.mean / self.rms.std), th.zeros_like(rewards)) * self.weight
            return std_rewards
        elif self.rwd_norm_type == "minmax":
            return (rewards - rewards.min()) / ((rewards.max() - rewards.min()) + 1e-8) * self.weight
        else:
            return rewards * self.weight
        
    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize the observations."""
        if self.obs_rms:
            x = (
                (
                    (x - self.obs_norm.mean.to(self.device))
                )
                    / th.sqrt(self.obs_norm.var.to(self.device))
            ).clip(-5, 5)
        else:
            x = x / 255.0 if len(self.obs_shape) > 2 else x
        return x

    def init_normalization(self, num_steps: int, num_iters: int, env: gym.Env, s) -> None:
        if self.obs_rms:
            next_ob = []
            print("Start to initialize observation normalization parameter.....")
            for step in tqdm(range(num_steps * num_iters)):
                acs = th.randint(0, env.action_space.n, size=(self.n_envs,))
                s, r, te, tr, _ = env.step(acs)
                next_ob += s.view(-1, *self.obs_shape).cpu()

                if len(next_ob) % (num_steps * self.n_envs) == 0:
                    next_ob = th.stack(next_ob).float()
                    self.obs_norm.update(next_ob)
                    next_ob = []

        # if "rms" in self.rwd_norm_type:
        #     print("Start to initialize reward normalization parameter.....")
        #     ob = []
        #     next_ob = []
        #     next_term = []
        #     next_trunc = []
        #     next_act = []
        #     for step in tqdm(range(num_steps)):
        #         acs = th.randint(0, env.action_space.n, size=(self.n_envs,))
        #         ob += s.unsqueeze(0)
        #         next_act += acs.unsqueeze(0)

        #         ns, r, te, tr, _ = env.step(acs)

        #         next_ob += ns.unsqueeze(0)
        #         next_term += te.unsqueeze(0)
        #         next_trunc += tr.unsqueeze(0)

        #         self.watch(
        #             observations=s,
        #             actions=acs,
        #             rewards=r,
        #             terminateds=te,
        #             truncateds=tr,
        #             next_observations=ns
        #         )

        #         s = ns

        #         if len(next_ob) % num_steps == 0:
        #             ob = th.stack(ob).float()
        #             next_ob = th.stack(next_ob).float()
        #             next_term = th.stack(next_term).float()
        #             next_trunc = th.stack(next_trunc).float()
        #             next_act = th.stack(next_act).float()

        #             samples = {
        #                 "observations": ob,
        #                 "actions": next_act,
        #                 "rewards": r,
        #                 "terminateds": next_term,
        #                 "truncateds": next_trunc,
        #                 "next_observations": next_ob
        #             }
        #             # this computes the rewards and also scales them
        #             next_rew = self.compute(samples)
        #             ob = []
        #             next_ob = []
        #             next_term = []
        #             next_trunc = []
        #             next_act = []
        return env

    @abstractmethod
    def watch(self, 
              observations: th.Tensor, 
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples, e.g., e.g., intrinsic rewards for the current samples. This 
            is useful when applying the memory-based methods to off-policy algorithms.
        """
    
    @abstractmethod
    def compute(self, samples: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors, whose keys are
            'observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'. For example, 
            the data shape of 'observations' is (n_steps, n_envs, *obs_shape). 

        Returns:
            The intrinsic rewards.
        """
        for key in ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations']:
            assert key in samples.keys(), f"Key {key} is not in samples."

        # update obs RMS if necessary
        if self.obs_rms:
            self.obs_norm.update(samples['observations'].reshape(-1, *self.obs_shape).cpu())

        self.global_step += 1

    @abstractmethod
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """