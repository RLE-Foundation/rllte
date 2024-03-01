import gymnasium as gym
import numpy as np

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