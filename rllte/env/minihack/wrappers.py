import gymnasium as gym
from IPython import embed


class Gym2Gymnasium(gym.Wrapper):
    def __init__(self, env):
        """Convert gym.Env to gymnasium.Env"""
        self.env = env

        orig_observation_space = env.observation_space
        new_observation_space = {}
        for k,v in orig_observation_space.spaces.items():
            if "Box" in str(type(v)):
                new_observation_space[k] = gym.spaces.Box(
                    low=v.low, high=v.high, shape=v.shape, dtype=v.dtype
                )
            
        self.observation_space = gym.spaces.Dict(new_observation_space)
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