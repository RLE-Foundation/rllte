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


from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.type_alias import VanillaReplayBatch
from rllte.xploit.storage.dict_replay_storage import DictReplayStorage


class HerReplayStorage(DictReplayStorage):
    """Hindsight experience replay (HER) storage for off-policy algorithms.
        Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py

    Args:
        observation_space (gym.Space): Observation space.
        action_space (gym.Space): Action space.
        device (str): Device to convert the data.
        storage_size (int): The capacity of the storage.
        num_envs (int): The number of parallel environments.
        batch_size (int): Batch size of samples.
        goal_selection_strategy (str): A goal selection strategy of ["future", "final", "episode"].
        num_goals (int): The number of goals to sample.
        reward_fn (Callable): Function to compute new rewards based on state and goal, whose definition is
            same as https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/bit_flipping_env.py#L190
        copy_info_dict (bool) whether to copy the info dictionary and pass it to compute_reward() method.

    Returns:
        Dict replay storage.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 device: str = "cpu",
                 storage_size: int = 1000000,
                 num_envs: int = 1,
                 batch_size: int = 1024,
                 goal_selection_strategy: str = "future",
                 num_goals: int = 4,
                 reward_fn: Optional[Callable] = None,
                 copy_info_dict: bool = False
                 ) -> None:
        super().__init__(observation_space, action_space, device, storage_size, batch_size, num_envs)

        assert goal_selection_strategy in ["future", "final", "episode", "random"]
        self.goal_selection_strategy = goal_selection_strategy
        self.num_goals = num_goals
        self.reward_fn = reward_fn
        self.copy_info_dict = copy_info_dict

        # compute ratio between HER replays and regular replays in percent
        self.her_ratio = 1 - (1.0 / (self.num_goals + 1))
        # store the info dict to compute the new reward
        self.infos = np.array([[{} for _ in range(self.num_envs)] for _ in range(self.storage_size)])
        # record the starting and end indices of each episode for creating virtual transitions
        self.ep_start = np.zeros((self.storage_size, self.num_envs), dtype=np.int64)
        self.ep_length = np.zeros((self.storage_size, self.num_envs), dtype=np.int64)
        self._current_ep_start = np.zeros(self.num_envs, dtype=np.int64)

        self.reset()
    
    def reset(self) -> None:
        """Reset the storage."""
        self.next_observations = {
            key: np.empty((self.storage_size, self.num_envs, *shape), dtype=self.observation_space[key].dtype)
            for key, shape in self.obs_shape.items()
        }

    def add(self,
            observations: Dict[str, th.Tensor],
            actions: th.Tensor,
            rewards: th.Tensor,
            terminateds: th.Tensor,
            truncateds: th.Tensor,
            infos: Dict[str, Any],
            next_observations: Dict[str, th.Tensor]
            ) -> None:
        """Add sampled transitions into storage.

        Args:
            observations (Dict[str, th.Tensor]): Observations.
            actions (th.Tensor): Actions.
            rewards (th.Tensor): Rewards.
            terminateds (th.Tensor): Termination flag.
            truncateds (th.Tensor): Truncation flag.
            infos (Dict[str, Any]): Additional information.
            next_observations (Dict[str, th.Tensor]): Next observations.

        Returns:
            None.
        """
        # set the length of the old episode to 0, so it can't be sampled anymore
        for env_idx in range(self.num_envs):
            episode_start = self.ep_start[self.step, env_idx]
            episode_length = self.ep_length[self.step, env_idx]
            if episode_length > 0:
                episode_end = episode_start + episode_length
                episode_indices = np.arange(self.step, episode_end) % self.buffer_size
                self.ep_length[episode_indices, env_idx] = 0

        # update episode start
        self.ep_start[self.step] = self._current_ep_start.copy()

        if self.copy_info_dict:
            self.infos[self.step] = infos
        # store the transition
        super().add(observations, actions, rewards, terminateds, truncateds, infos, next_observations)

        # compute and store the episode length when episode ends
        for env_idx in range(self.num_envs):
            if terminateds[env_idx].item() or truncateds[env_idx].item():
                self._compute_episode_length(env_idx)

    def _compute_episode_length(self, env_idx: int) -> None:
        """Compute and store the episode length for environment with index env_idx.

        Args:
            env_idx (int): Index of the environment.

        Returns:
            None.
        """
        episode_start = self._current_ep_start[env_idx]
        episode_end = self.step
        if episode_end < episode_start:
            # occurs when the buffer becomes full, the storage resumes at the beginning of the buffer.
            episode_end += self.storage_size
        episode_indices = np.arange(episode_start, episode_end) % self.storage_size
        self.ep_length[episode_indices, env_idx] = episode_end - episode_start
        # update the current episode start
        self._current_ep_start[env_idx] = self.step

    def sample(self) -> VanillaReplayBatch:
        """Sample from the storage."""
        # check if we have complete episodes
        is_valid = self.ep_length > 0
        if not np.any(is_valid):
            raise RuntimeError(
                "Unable to sample before the end of the first episode. We recommend choosing a value "
                "for num_init_steps that is greater than the maximum number of timesteps in the environment."
            )

        # get vaild indices
        valid_indices = np.flatnonzero(is_valid)
        sampled_indices = np.random.choice(valid_indices, size=self.batch_size, replace=True)
        batch_indices, env_indices = np.unravel_index(sampled_indices, is_valid.shape)

        # split the indexes between real and virtual transitions.
        nb_virtual = int(self.her_ratio * self.batch_size)
        virtual_batch_indices, real_batch_indices = np.split(batch_indices, [nb_virtual])
        virtual_env_indices, real_env_indices = np.split(env_indices, [nb_virtual])

        # get real and virtual samples
        real_data = self._sample_real(real_batch_indices, real_env_indices)
        # create virtual transitions by sampling new desired goals and computing new rewards
        virtual_data = self._sample_virtual(virtual_batch_indices, virtual_env_indices)

        # concatenate real and virtual data
        observations = {
            key: th.cat((real_data.observations[key], virtual_data.observations[key]))
            for key in virtual_data.observations.keys()
        }
        actions = th.cat((real_data.actions, virtual_data.actions))
        next_observations = {
            key: th.cat((real_data.next_observations[key], virtual_data.next_observations[key]))
            for key in virtual_data.next_observations.keys()
        }
        terminateds = th.cat((real_data.terminateds, virtual_data.terminateds))
        truncateds = th.cat((real_data.truncateds, virtual_data.truncateds))
        rewards = th.cat((real_data.rewards, virtual_data.rewards))

        return VanillaReplayBatch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            next_observations=next_observations,
        )

    def _sample_real(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> VanillaReplayBatch:
        """Get real samples from the storage.

        Args:
            batch_indices (np.ndarray): Batch indices of transitions.
            env_indices (np.ndarray): Environment indices.

        Returns:
            Batched samples.
        """
        # get real samples
        obs = {key: item[batch_indices, env_indices, :] for key, item in self.observations.items()}
        next_obs = {key: item[batch_indices, env_indices, :] for key, item in self.next_observations.items()}

        actions = self.actions[batch_indices, env_indices]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1)
        terminateds = self.terminateds[batch_indices, env_indices].reshape(-1, 1)
        truncateds = self.truncateds[batch_indices, env_indices].reshape(-1, 1)

        # convert to torch tensor
        observations = {key: self.to_torch(item) for key, item in obs.items()}
        next_observations = {key: self.to_torch(item) for key, item in next_obs.items()}

        return VanillaReplayBatch(
            observations=observations,
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            terminateds=self.to_torch(terminateds),
            truncateds=self.to_torch(truncateds),
            next_observations=next_observations,
        )

    def _sample_virtual(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> VanillaReplayBatch:
        """Get the samples with new goals and rewards.

        Args:
            batch_indices (np.ndarray): Batch indices of transitions.
            env_indices (np.ndarray): Environment indices.

        Returns:
            Batched samples.
        """
        # Get real samples
        obs = {key: item[batch_indices, env_indices, :] for key, item in self.observations.items()}
        next_obs = {key: item[batch_indices, env_indices, :] for key, item in self.next_observations.items()}

        actions = self.actions[batch_indices, env_indices]
        rewards = self.rewards[batch_indices, env_indices].reshape(-1, 1)
        terminateds = self.terminateds[batch_indices, env_indices].reshape(-1, 1)
        truncateds = self.truncateds[batch_indices, env_indices].reshape(-1, 1)

        # copy infos if needed
        if self.copy_info_dict:
            infos = deepcopy(self.infos[batch_indices, env_indices])
        else:
            infos = [{} for _ in range(len(batch_indices))]
        # sample and set new goals
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        # the desired goal for the next observation must be the same as the previous one
        next_obs["desired_goal"] = new_goals

        # compute new rewards
        rewards = self.reward_fn(
            # the new state depends on the previous state and action
            # s_{t+1} = f(s_t, a_t)
            # so the next achieved_goal depends also on the previous state and action
            # because we are in a GoalEnv:
            # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
            # therefore we have to use next_obs["achieved_goal"] and not obs["achieved_goal"]
            next_obs["achieved_goal"],
            # here we use the new desired goal
            obs["desired_goal"],
            infos,
        )

        # convert to torch tensor
        observations = {key: self.to_torch(item) for key, item in obs.items()}
        next_observations = {key: self.to_torch(item) for key, item in next_obs.items()}

        return VanillaReplayBatch(
            observations=observations,
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            terminateds=self.to_torch(terminateds),
            truncateds=self.to_torch(truncateds),
            next_observations=next_observations,
        )

    def _sample_goals(self, batch_indices: np.ndarray, env_indices: np.ndarray) -> np.ndarray:
        """Sample goals based on goal_selection_strategy.

        Args:
            batch_indices (np.ndarray): Batch indices of transitions.
            env_indices (np.ndarray): Environment indices.

        Returns:
            Sampled goals.
        """
        batch_ep_start = self.ep_start[batch_indices, env_indices]
        batch_ep_length = self.ep_length[batch_indices, env_indices]

        if self.goal_selection_strategy == "final":
            # replay with final state of current episode
            transition_indices_in_episode = batch_ep_length - 1

        elif self.goal_selection_strategy == "future":
            # replay with random state which comes from the same episode and was observed after current transition
            current_indices_in_episode = (batch_indices - batch_ep_start) % self.storage_size
            transition_indices_in_episode = np.random.randint(current_indices_in_episode, batch_ep_length)

        elif self.goal_selection_strategy == "episode":
            # replay with random state which comes from the same episode as current transition
            transition_indices_in_episode = np.random.randint(0, batch_ep_length)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        transition_indices = (transition_indices_in_episode + batch_ep_start) % self.storage_size
        return self.next_observations["achieved_goal"][transition_indices, env_indices]

    def update(self, *args, **kwargs) -> None:
        """Update the storage if necessary."""
        return None
