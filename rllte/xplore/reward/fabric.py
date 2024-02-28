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


from typing import Dict, Optional

import torch as th
from rllte.common.prototype import BaseReward


class Fabric(BaseReward):
    """Connecting multiple intrinsic reward modules to generate mixed intrinsic rewards.

    Args:
        rewards (BaseReward): A series of intrinsic reward modules.

    Returns:
        Instance of RE3.
    """

    def __init__(self, *rewards: BaseReward) -> None:
        for rwd in rewards:
            assert isinstance(
                rwd, BaseReward
            ), "The input rewards must be the instance of BaseReward!"
        self.rewards = list(rewards)

    def watch(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_observations: th.Tensor,
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
            Feedbacks for the current samples.
        """
        for rwd in self.rewards:
            rwd.watch(
                observations,
                actions,
                rewards,
                terminateds,
                truncateds,
                next_observations,
            )

    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        intrinsic_rewards = []
        for rwd in self.rewards:
            intrinsic_rewards.append(rwd.compute(samples, sync))

        return intrinsic_rewards

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        for rwd in self.rewards:
            rwd.update(samples)