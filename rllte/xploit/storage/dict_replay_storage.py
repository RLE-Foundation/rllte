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


from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_storage import VanillaReplayBatch
from rllte.xploit.storage.vanilla_replay_storage import VanillaReplayStorage

class DictReplayStorage(VanillaReplayStorage):
    """Dict replay storage for off-policy algorithms and dictionary observations.

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to store the data.
        storage_size (int): Storage size.
        num_envs (int): The number of parallel environments.
        batch_size (int): Batch size.

    Returns:
        Dict replay storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 1000000,
        num_envs: int = 1,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, num_envs, batch_size)

        assert isinstance(self.obs_shape, dict), "DictReplayStorage only support Dict observation space."
        
        # data containers
        ###########################################################################################################
        self.observations = {
            key: np.empty((self.storage_size, num_envs, *shape), dtype=observation_space[key].dtype) 
            for key, shape in self.obs_shape.items()
        }    
        ###########################################################################################################

    def add(
        self,
        observations: Dict[str, th.Tensor],
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        info: Dict[str, Any],
        next_observations: Dict[str, th.Tensor],
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            observations (Dict[str, th.Tensor]): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Termination flag.
            truncateds (th.Tensor): Truncation flag.
            info (Dict[str, Any]): Additional information.
            next_observations (Dict[str, th.Tensor]): Next observations.

        Returns:
            None.
        """
        for key in self.observations.keys():
            # broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], gym.spaces.Discrete):
                obs_ = observations[key].reshape((self.num_envs,) + self.obs_shape[key])
                next_obs_ = next_observations[key].reshape((self.num_envs,) + self.obs_shape[key])
            else:
                obs_ = observations[key]
                next_obs_ = next_observations[key]
            
            np.copyto(self.observations[key][self.step], obs_.cpu().numpy())
            np.copyto(self.observations[key][(self.step + 1) % self.storage_size], next_obs_.cpu().numpy())

        np.copyto(self.actions[self.step], actions.cpu().numpy())
        np.copyto(self.rewards[self.step], rewards.cpu().numpy())
        np.copyto(self.terminateds[self.step], terminateds.cpu().numpy())
        np.copyto(self.truncateds[self.step], truncateds.cpu().numpy())

        self.step = (self.step + 1) % self.storage_size
        self.full = self.full or self.step == 0

    def sample(self, step: int) -> VanillaReplayBatch:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        # get batch and env indices
        if self.full:
            batch_indices = (np.random.randint(1, self.storage_size, size=self.batch_size) + self.step) % self.storage_size
        else:
            batch_indices = np.random.randint(0, self.step, size=self.batch_size)
        env_indices = np.random.randint(0, self.num_envs, size=(self.batch_size, ))

        # get batch data
        obs = {self.observations[key][batch_indices, env_indices, :] for key in self.observations.keys()}
        next_obs = {self.observations[key][(batch_indices + 1) % self.storage_size, env_indices, :] for key in self.observations.keys()}

        actions = self.actions[batch_indices, env_indices]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1)
        terminateds = self.terminateds[batch_indices, env_indices].reshape(-1, 1)
        truncateds = self.truncateds[batch_indices, env_indices].reshape(-1, 1)

        # convert to torch tensors
        observations = {key: self.to_torch(item) for key, item in obs.items()}
        next_observations = {key: self.to_torch(item) for key, item in next_obs.items()}

        return VanillaReplayBatch(
            observations=observations,
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            terminateds=self.to_torch(terminateds),
            truncateds=self.to_torch(truncateds),
            next_observations=next_observations
        )

    def update(self, *args) -> None:
        """Update the storage if necessary."""
        return None