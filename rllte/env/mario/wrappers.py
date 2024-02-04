import gymnasium as gym
import numpy as np

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game
        over.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.env = env

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = np.logical_or(terminated, truncated)
        lives = self.env.unwrapped.env._life
        if self.lives > lives > 0:
            terminated, truncated = True, True
        self.lives = lives
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.env._life
        return obs
    
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip
        self.env = env

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if np.logical_or(terminated, truncated):
                break
        return obs, total_reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        return self.env.reset()
    
    def render(self):
        return self.env.render()


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