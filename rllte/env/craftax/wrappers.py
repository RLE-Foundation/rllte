import jax
import gymnasium as gym
import torch
from dataclasses import asdict
from brax.io import torch as brax_torch

from typing import Any, Dict, Tuple, Optional
import numpy as np

class TorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""
    
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device
        self.env = env
        self.default_params = env.default_params
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
        }
        
        # define obs and action space
        obs_shape = env.observation_space(self.default_params).shape
        self.observation_space = gym.spaces.Box(
            low=-1e6, high=1e6, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(env.action_space(self.default_params).n)

        # jit the reset function
        def reset(key):
            key1, key2 = jax.random.split(key)
            obs, state = self.env.reset(key2)
            return state, obs, key1, asdict(state)
        self._reset = jax.jit(reset)

        # jit the step function
        def step(state, action):
            obs, env_state, reward, done, info = self.env.step(rng=self._key, state=state, action=action)
            return env_state, obs, reward, done, {**asdict(env_state), **info}
        self._step = jax.jit(step)

    def reset(self, seed=0, options=None):
        self.seed(seed)
        self._state, obs, self._key, info = self._reset(self._key)
        return brax_torch.jax_to_torch(obs, device=self.device), info

    def step(self, action):
        action = brax_torch.torch_to_jax(action)
        self._state, obs, reward, done, info = self._step(self._state, action)
        obs = brax_torch.jax_to_torch(obs, device=self.device)
        reward = brax_torch.jax_to_torch(reward, device=self.device)
        terminateds = brax_torch.jax_to_torch(done, device=self.device)
        truncateds = brax_torch.jax_to_torch(done, device=self.device)
        info = brax_torch.jax_to_torch(info, device=self.device)
        return obs, reward, terminateds, truncateds, info

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

class ResizeTorchWrapper(gym.Wrapper):
    """Wrapper that resizes observations to a given shape."""
    
    def __init__(self, env, shape):
        super().__init__(env)
        self.env = env
        num_channels = env.observation_space.shape[-1]
        self.shape = (num_channels, shape[0], shape[1])

        # define obs and action space
        self.observation_space = gym.spaces.Box(
            low=-1e6, high=1e6, shape=self.shape)
        
    def reset(self, seed=0, options=None):
        obs, info = self.env.reset(seed, options)
        obs = obs.permute(0, 3, 1, 2)
        obs = torch.nn.functional.interpolate(obs, size=self.shape[1:], mode='nearest')
        return obs, info
    
    def step(self, action):
        obs, reward, terminateds, truncateds, info = self.env.step(action)
        obs = obs.permute(0, 3, 1, 2)
        obs = torch.nn.functional.interpolate(obs, size=self.shape[1:], mode='nearest')
        return obs, reward, terminateds, truncateds, info
    
class RecordEpisodeStatistics4Craftax(gym.Wrapper):
    """Keep track of cumulative rewards and episode lengths. 
    This wrapper is dedicated to EnvPool-based Atari games.

    Args:
        env (gym.Env): Environment to wrap.
        deque_size (int): The size of the buffers :attr:`return_queue` and :attr:`length_queue`
    
    Returns:
        RecordEpisodeStatistics4EnvPool instance.
    """
    def __init__(self, env: gym.Env, deque_size: int = 100) -> None:
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
    
    def reset(self, **kwargs):
        observations, infos = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos
    
    def step(self, actions):
        observations, rewards, terms, truncs, infos = super().step(actions)
        self.episode_returns += rewards.cpu().numpy()
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["returned_episode"].cpu().numpy().astype(np.int32)
        self.episode_lengths *= 1 - infos["returned_episode"].cpu().numpy().astype(np.int32)
        infos["episode"] = {}
        infos["episode"]["r"] = self.returned_episode_returns
        infos["episode"]["l"] = self.returned_episode_lengths

        for idx, d in enumerate(terms):
            if not d:
                infos["episode"]["r"][idx] = 0
                infos["episode"]["l"][idx] = 0

        return observations, rewards, terms, truncs, infos