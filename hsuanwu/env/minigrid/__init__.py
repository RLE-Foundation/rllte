import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper

from hsuanwu.env.utils import FrameStack, TorchVecEnvWrapper


class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space["image"]
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[2], shape[0], shape[1]),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return np.transpose(observation["image"], axes=[2, 0, 1])


def make_minigrid_env(
    env_id: str = "Alien-v5",
    num_envs: int = 8,
    fully_observable: bool = True,
    seed: int = 0,
    frame_stack: int = 1,
    device: str = "cpu",
    distributed: bool = False,
) -> gym.Env:
    """Build MiniGrid environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        fully_observable (bool): 'True' for using fully observable RGB image as observation.
        seed (int): Random seed.
        frame_stack (int): Number of stacked frames.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        distributed (bool): For `Distributed` algorithms, in which `SyncVectorEnv` is required
            and reward clip will be used before environment vectorization.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> gym.Env:
        def _thunk():
            env = gym.make(env_id)

            if fully_observable:
                env = FullyObsWrapper(env)
                env = Minigrid2Image(env)
                env = FrameStack(env, k=frame_stack)
            else:
                env = FlatObsWrapper(env)

            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            return env

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    if distributed:
        envs = SyncVectorEnv(envs)
    else:
        envs = AsyncVectorEnv(envs)
        envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device=device)
