import gym as old_gym
import gymnasium as gym
import numpy as np
from griddly import GymWrapperFactory, gd
import os
import torch as th

def register_griddly_envs():
    env_dict = old_gym.envs.registration.registry.copy()
    for env in env_dict:
        if 'MazeEnv' in env:
            print("Remove {} from registry".format(env))
            del old_gym.envs.registration.registry[env]
    wrapper = GymWrapperFactory()
    wrapper.build_gym_from_yaml('MazeEnv', f"{os.getcwd()}/rllte/env/griddly/maze_env.yaml")

class GymnasiumGriddlyEnv(gym.Env):
    def __init__(self, env, obs_shape, max_steps=500, episodic=False):
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)

        obs_, _ = self.env.reset()
        self.episodic_heatmap = np.zeros((obs_.shape[1], obs_.shape[2]), dtype=np.int8)
        self.global_heatmap = np.zeros((obs_.shape[1], obs_.shape[2]), dtype=np.int8)
        
        self.max_steps = max_steps
        self.episodic = episodic

    def init_heatmaps(self, obs):
        self.episodic_heatmap[obs[3] == 1] = -1
        self.global_heatmap[obs[3] == 1] = -1

    def step(self, action):
        obs, reward, te, tr, info = self.env.step(action)
        img = self.env.unwrapped.env.render(observer="global", mode="rgb_array")
        
        # add to heatmaps
        agent_pos = np.unravel_index(np.argmax(obs[0]), obs[0].shape)
        self.episodic_heatmap[agent_pos] += 1
        self.global_heatmap += self.episodic_heatmap
        if self.t >= self.max_steps:
            te = True
            tr = True
            if self.episodic:
                reward = np.sum(self.episodic_heatmap > 0) / (np.prod(self.episodic_heatmap.shape) - np.sum(self.episodic_heatmap == -1))
            else:
                reward = np.sum(self.global_heatmap > 0) / (np.prod(self.global_heatmap.shape) - np.sum(self.global_heatmap == -1))
        self.t += 1
        return img, reward, te, tr, info

    def reset(self, options=None, seed=None):
        self.t = 0
        obs, info = self.env.reset(options=options, seed=seed)

        # init heatmaps        
        self.episodic_heatmap = np.zeros_like(self.episodic_heatmap)
        self.init_heatmaps(obs)
        
        img = self.env.unwrapped.env.render(observer="global", mode="rgb_array")
        return img, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

class Gym2Gymnasium(gym.Wrapper):
    def __init__(self, env):
        """Convert gym.Env to gymnasium.Env"""
        self.env = env

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)

    def step(self, action):
        """Repeat action, and sum reward"""
        return self.env.step(action)

    def reset(self, options=None, seed=None):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)
    
class ImageTranspose(gym.ObservationWrapper):
    """Transpose observation from channels last to channels first.

    Args:
        env (gym.Env): Environment to wrap.

    Returns:
        Minigrid2Image instance.
    """

    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shape = env.observation_space.shape
        dtype = env.observation_space.dtype
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )

    def observation(self, observation):
        """Convert observation to image."""
        observation= np.transpose(observation, axes=[2, 0, 1])
        return observation
