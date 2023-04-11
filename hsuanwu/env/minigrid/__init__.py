import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper

from hsuanwu.common.typing import Any, Device, Dict, Env, Ndarray, Tensor, Tuple
from hsuanwu.env.utils import FrameStack


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


class TorchVecEnvWrapper(gym.Wrapper):
    """Build environments that output torch tensors.

    Args:
        env (Env): The environment.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        TorchVecEnv instance.
    """

    def __init__(self, env: Env, device: Device) -> None:
        super().__init__(env)
        self._device = torch.device(device)
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = len(env.envs)

    def reset(self, **kwargs) -> Tuple[Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        obs = torch.as_tensor(obs, device=self._device)
        return obs, info

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(
            action.squeeze(1).cpu().numpy()
        )
        obs = torch.as_tensor(obs, device=self._device)
        reward = torch.as_tensor(
            reward, dtype=torch.float32, device=self._device
        ).unsqueeze(dim=1)
        terminated = torch.as_tensor(
            [[1.0] if _ else [0.0] for _ in terminated],
            dtype=torch.float32,
            device=self._device,
        )
        truncated = torch.as_tensor(
            [[1.0] if _ else [0.0] for _ in truncated],
            dtype=torch.float32,
            device=self._device,
        )

        return obs, reward, terminated, truncated, info


def make_minigrid_env(
    env_id: str = "Alien-v5",
    num_envs: int = 8,
    fully_observable: bool = True,
    seed: int = 0,
    frame_stack: int = 1,
    device: torch.device = "cuda",
) -> Env:
    """Build MiniGrid environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of parallel environments.
        fully_observable (bool): 'True' for using fully observable RGB image as observation.
        seed (int): Random seed.
        frame_stack (int): Number of stacked frames.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> Env:
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
    envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device=device)
