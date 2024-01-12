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


from typing import Dict, List
import gymnasium as gym
import torch as th

from base_reward import BaseReward
from model import ObservationEncoder

class REVD(BaseReward):
    """Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning (REVD).
        See paper: https://openreview.net/pdf?id=V2pw1VYMrDo

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        n_envs (int): The number of parallel environments.
        episode_length (int): The maximum length of an episode.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        use_rms (bool): Use running mean and std for normalization.
        latent_dim (int): The dimension of encoding vectors.
        alpha (alpha): The The order of Rényi entropy.
        k (int): Use the k-th neighbors.
        average_divergence (bool): Use the average of divergence estimation.

    Returns:
        Instance of RISE.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        episode_length: int,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        use_rms: bool = True,
        latent_dim: int = 128,
        alpha: float = 0.5,
        k: int = 5,
        average_divergence: bool = False
        ) -> None:
        super().__init__(observation_space, action_space, n_envs, device, beta, kappa, use_rms)
        
        # build the storage for random embeddings
        self.storage_size = episode_length
        self.storage = th.zeros(size=(episode_length, n_envs, latent_dim))
        # set parameters
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.k = k
        self.average_divergence = average_divergence
        # build the random encoder and freeze the network parameters
        self.random_encoder = ObservationEncoder(obs_shape=self.obs_shape, latent_dim=latent_dim).to(self.device)
        for p in self.random_encoder.parameters():
            p.requires_grad = False
        
        self.first_update = True
        self.last_encoded_obs: List = list()
    
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
        assert n_steps == self.storage_size, "REVD must be invoked after the episode is finished!"
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)

        # first update to fill in the storage
        if self.first_update:
            with th.no_grad():
                for i in range(n_envs):
                    self.storage_last[:, i] = self.random_encoder(obs_tensor[:, i])
            self.first_update = False
            return intrinsic_rewards
        # REVD requires at least two episodes
        with th.no_grad():
            for i in range(n_envs):
                # get the target features
                tgt_feats = self.storage[:, i]
                # get the source features
                src_feats = self.random_encoder(obs_tensor[:, i])
                # compute the intra and outer distances
                dist_intra = th.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                dist_outer = th.linalg.vector_norm(src_feats.unsqueeze(1) - tgt_feats, ord=2, dim=2)
                # compute the divergence with average estimation
                if self.average_divergence:
                    for sub_k in range(self.k):
                        # L = th.kthvalue(dist_intra, 2, dim=1).values.sum() / n_steps
                        D_step_intra = th.kthvalue(dist_intra, sub_k + 1, dim=1).values
                        D_step_outer = th.kthvalue(dist_outer, sub_k + 1, dim=1).values
                        intrinsic_rewards[:, i] += th.pow(D_step_outer / (D_step_intra + 1e-11), 1.0 - self.alpha)
                else:
                    D_step_intra = th.kthvalue(dist_intra, self.k + 1, dim=1).values
                    D_step_outer = th.kthvalue(dist_outer, self.k + 1, dim=1).values
                    # L = th.kthvalue(dist_intra, 2, dim=1).values.sum() / num_steps
                    intrinsic_rewards[:, i] = th.pow(D_step_outer / (D_step_intra + 1e-11), 1.0 - self.alpha)
                
                # save the observations of the last episode
                self.storage[:, i] = src_feats
        
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