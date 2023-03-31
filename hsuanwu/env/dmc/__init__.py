from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register

from hsuanwu.common.typing import Dict, Tensor, Device, Tuple, List, Env, Any, Ndarray


class FrameStackEnv(gym.Wrapper):
    """Observation wrapper that stacks the observations in a rolling manner.
    
    Args:
        env (Env): Environment to wrap.
        k: Number of stacked frames.
    
    Returns:
        FrameStackEnv instance.
    """

    def __init__(self, env: Env, k: int) -> None:
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self, **kwargs) -> Tuple[Tensor, Dict]:
        obs, info = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: Tuple[float]) -> Tuple[Any, float, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> Ndarray:
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

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
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs) -> Tuple[Tensor, Dict]:
        obs, info = self.env.reset(**kwargs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        return obs, info

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor, Tensor, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self._device)
        terminated = torch.as_tensor(
            1.0 if terminated else 0.0,
            dtype=torch.float32,
            device=self._device,
        )

        return obs, reward, terminated, truncated, info

def make_dmc_env(
    env_id: str = "cartpole_balance",
    device: Device = 'cuda', 
    resource_files: List = None,
    img_source: str = None,
    total_frames: int = None,
    seed: int = 1,
    visualize_reward: bool = False,
    from_pixels: bool = True,
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_stack: int = 3,
    frame_skip: int = 2,
    episode_length: int = 1000,
    environment_kwargs: Dict = None,
):
    """Build DeepMind Control Suite environments.

    Args:
        env_id (str): Name of environment.
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        resource_files (List): File path of the resource files.
        img_source (str): Type of the background distractor, supported values: ['color', 'noise', 'images', 'video'].
        total_frames (int): for 'images' or 'video' distractor.
        seed (int): Random seed.
        visualize_reward (bool): True when 'from_pixels' is False, False when 'from_pixels' is True.
        from_pixels (bool): Provide image-based observations or not.
        height (int): Image observation height.
        width (int): Image observation width.
        camera_id (int): Camera id for generating image-based observations.
        frame_stack (int): Number of stacked frames.
        frame_skip (int): Number of action repeat.
        episode_length (int): Maximum length of an episode.
        environment_kwargs (Dict): Other environment arguments.

    Returns:
        Environments instance.
    """
    domain_name, task_name = env_id.split("_")
    env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry:
        register(
            id=env_id,
            entry_point="hsuanwu.env.dmc.wrappers:DMCWrapper",
            kwargs={
                "domain_name": domain_name,
                "task_name": task_name,
                "resource_files": resource_files,
                "img_source": img_source,
                "total_frames": total_frames,
                "task_kwargs": {"random": seed},
                "environment_kwargs": environment_kwargs,
                "visualize_reward": visualize_reward,
                "from_pixels": from_pixels,
                "height": height,
                "width": width,
                "camera_id": camera_id,
                "frame_skip": frame_skip,
            },
            max_episode_steps=max_episode_steps,
        )
    if visualize_reward:
        return TorchVecEnvWrapper(gym.make(env_id), device)
    else:
        env = FrameStackEnv(gym.make(env_id), frame_stack)
        return TorchVecEnvWrapper(env, device)
