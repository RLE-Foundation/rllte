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


from typing import Dict
import gymnasium as gym
import torch as th

from base_reward import BaseReward
from model import CnnObservationEncoder, MlpObservationEncoder

from rllte.common.preprocessing import is_image_space

class RE3(BaseReward):
    """State Entropy Maximization with Random Encoders for Efficient Exploration (RE3).
        See paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        use_rms (bool): Use running mean and std for normalization.
        latent_dim (int): The dimension of encoding vectors.
        storage_size (int): The size of the storage for random embeddings.
        n_envs (int): The number of parallel environments.
        k (int): Use the k-th neighbors.
        average_entropy (bool): Use the average of entropy estimation.

    Returns:
        Instance of RE3.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 1.0,
        use_rms: bool = True,
        latent_dim: int = 128,
        storage_size: int = 10000,
        k: int = 5,
        average_entropy: bool = False
        ) -> None:
        super().__init__(observation_space, action_space, device, beta, kappa, use_rms)
        
        # build the storage for random embeddings
        self.storage_size = storage_size // n_envs
        self.storage = th.zeros(size=(self.storage_size, n_envs, latent_dim))
        self.storage_idx = 0
        self.storage_full = False
        # set parameters
        self.latent_dim = latent_dim
        self.k = k
        self.average_entropy = average_entropy
        # build the random encoder and freeze the network parameters
        if is_image_space(observation_space):
            self.random_encoder = CnnObservationEncoder(obs_shape=self.obs_shape, 
                                                        latent_dim=latent_dim).to(self.device)
        else:
            self.random_encoder = MlpObservationEncoder(obs_shape=self.obs_shape,
                                                        latent_dim=latent_dim).to(self.device)
        for p in self.random_encoder.parameters():
            p.requires_grad = False
    
    def watch(self, 
              observations: th.Tensor,
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> None:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): The observations data with shape (n_steps, n_envs, *obs_shape).
            actions (th.Tensor): The actions data with shape (n_steps, n_envs, *action_shape).
            rewards (th.Tensor): The rewards data with shape (n_steps, n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_steps, n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_steps, n_envs).
            next_observations (th.Tensor): The next observations data with shape (n_steps, n_envs, *obs_shape).

        Returns:
            None.
        """
        with th.no_grad():
            self.storage[self.storage_idx] = self.random_encoder(observations)
            self.storage_idx = (self.storage_idx + 1) % self.storage_size

        # update the storage status
        self.storage_full = self.storage_full or self.storage_idx == 0

    def compute(self, samples: Dict) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict): The collected samples. A python dict like
                {observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                next_observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
                The derived intrinsic rewards have the shape of (n_steps, n_envs).

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)
        # get the number of steps and environments
        assert "observations" in samples.keys(), "The key `observations` must be contained in samples!"
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        obs_tensor = samples.get("observations").to(self.device)

        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            # get the target features
            tgt_feats = self.storage[: self.storage_idx].view(-1, self.latent_dim) if not self.storage_full \
                else self.storage.view(-1, self.latent_dim)
            # get the source features
            src_feats = self.random_encoder(obs_tensor.view(-1, *self.obs_shape))
            # compute the distance
            dist = th.linalg.vector_norm(src_feats.unsqueeze(1) - tgt_feats.to(self.device), ord=2, dim=2)
            # compute the entropy with average estimation
            if self.average_entropy:
                for sub_k in range(self.k):
                    intrinsic_rewards = th.log(th.kthvalue(dist, sub_k + 1, dim=1).values + 1.0)
                intrinsic_rewards /= self.k
            else:
                intrinsic_rewards = th.log(th.kthvalue(dist, self.k + 1, dim=1).values + 1.0)
            # reshape the intrinsic rewards
            intrinsic_rewards = intrinsic_rewards.view(n_steps, n_envs)

        # scale the intrinsic rewards
        return self.scale(intrinsic_rewards)
    
    def update(self, samples: Dict) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict): The collected samples. A python dict like
                {observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                next_observations (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """
        raise NotImplementedError