from typing import Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import (
    FrameStack,
    GrayScaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    TransformReward,
)

from hsuanwu.env.atari.wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


class TorchVecEnvWrapper(gym.Wrapper):
    """Build environments that output torch tensors.

    Args:
        env (Env): The environment.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.

    Returns:
        TorchVecEnv instance.
    """

    def __init__(self, env: gym.Env, device: th.Device) -> None:
        super().__init__(env)
        self._device = th.device(device)
        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.num_envs = len(env.envs)

    def reset(self, **kwargs) -> Tuple[th.Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        obs = th.as_tensor(obs, device=self._device)
        return obs, info

    def step(self, action: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(
            action.squeeze(1).cpu().numpy()
        )
        obs = th.as_tensor(obs, device=self._device)
        reward = th.as_tensor(
            reward, dtype=th.float32, device=self._device
        ).unsqueeze(dim=1)
        terminated = th.as_tensor(
            [[1.0] if _ else [0.0] for _ in terminated],
            dtype=th.float32,
            device=self._device,
        )
        truncated = th.as_tensor(
            [[1.0] if _ else [0.0] for _ in truncated],
            dtype=th.float32,
            device=self._device,
        )

        return obs, reward, terminated, truncated, info


def make_atari_env(
    env_id: str = "Alien-v5",
    num_envs: int = 8,
    device: th.device = "cpu",
    seed: int = 0,
    frame_stack: int = 4,
) -> gym.Env:
    """Build Atari environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of parallel environments.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        seed (int): Random seed.
        frame_stack (int): Number of stacked frames.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            env = gym.make(env_id)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=frame_stack)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)

            env = TransformReward(env, lambda reward: np.sign(reward))
            env = ResizeObservation(env, shape=(84, 84))
            env = GrayScaleObservation(env)
            env = FrameStack(env, frame_stack)

            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            return env

        return _thunk

    if "NoFrameskip-v4" not in env_id:
        env_id = "ALE/" + env_id
    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = SyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device)
