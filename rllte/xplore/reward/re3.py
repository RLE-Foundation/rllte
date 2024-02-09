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
import gymnasium as gym
import torch as th

from rllte.common.prototype import BaseReward
from .model import ObservationEncoder

class RE3(BaseReward):
    """State Entropy Maximization with Random Encoders for Efficient Exploration (RE3).
        See paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        n_envs (int): The number of parallel environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        rwd_norm_type (bool): Use running mean and std for reward normalization.
        obs_rms (bool): Use running mean and std for observation normalization.
        gamma (Optional[float]): Intrinsic reward discount rate, None for no discount.
        latent_dim (int): The dimension of encoding vectors.
        storage_size (int): The size of the storage for random embeddings.
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
        kappa: float = 0.0,
        rwd_norm_type: str = "rms",
        obs_rms: bool = True,
        gamma: Optional[float] = None,
        latent_dim: int = 128,
        storage_size: int = 1000,
        k: int = 5,
        average_entropy: bool = False,
        encoder_model: str = "mnih",
        weight_init: str = "default"
        ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, rwd_norm_type, obs_rms, gamma)
        
        # build the storage for random embeddings
        self.storage_size = storage_size
        self.storage = th.zeros(size=(storage_size, n_envs, latent_dim))
        self.storage_idx = 0
        self.storage_full = False
        # set parameters
        self.latent_dim = latent_dim
        self.k = k
        self.average_entropy = average_entropy
        # build the random encoder and freeze the network parameters
        self.random_encoder = ObservationEncoder(obs_shape=self.obs_shape, latent_dim=latent_dim, encoder_model=encoder_model, weight_init=weight_init).to(self.device)
        for p in self.random_encoder.parameters():
            p.requires_grad = False
    
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
            Feedbacks for the current samples, e.g., intrinsic rewards for the current samples. This 
            is useful when applying the memory-based methods to off-policy algorithms.
        """
        with th.no_grad():
            observations = self.normalize(observations)
            self.storage[self.storage_idx] = self.random_encoder(observations)
            self.storage_idx = (self.storage_idx + 1) % self.storage_size

        # update the storage status
        self.storage_full = self.storage_full or self.storage_idx == 0

    def compute(self, samples: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors, whose keys are
            'observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'. For example, 
            the data shape of 'observations' is (n_steps, n_envs, *obs_shape). 

        Returns:
            The intrinsic rewards.
        """
        super().compute(samples)
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        obs_tensor = samples.get("observations").to(self.device)
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            for i in range(n_envs):
                # get the target features
                tgt_feats = self.storage[:self.storage_idx, i] if not self.storage_full else self.storage[:, i]
                # get the source features
                src_feats = self.random_encoder(obs_tensor[:, i])
                # compute the distance
                dist = th.linalg.vector_norm(src_feats.unsqueeze(1) - tgt_feats.to(self.device), ord=2, dim=2)
                # compute the entropy with average estimation
                if self.average_entropy:
                    for sub_k in range(self.k):
                        intrinsic_rewards[:, i] += th.log(th.kthvalue(dist, sub_k + 1, dim=1).values + 1.0)
                    intrinsic_rewards[:, i] /= self.k
                else:
                    intrinsic_rewards[:, i] = th.log(th.kthvalue(dist, self.k + 1, dim=1).values + 1.0)
        
        # scale the intrinsic rewards
        return self.scale(intrinsic_rewards)
    
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """
        raise NotImplementedError