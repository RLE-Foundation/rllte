from typing import Callable, Dict, List, Optional

import gymnasium as gym
from gymnasium.envs import registry
from gymnasium.envs.registration import register
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

from hsuanwu.env.utils import FrameStack, TorchVecEnvWrapper


def make_dmc_env(
    env_id: str = "cartpole_balance",
    num_envs: int = 1,
    device: str = "cpu",
    resource_files: Optional[List] = None,
    img_source: Optional[str] = None,
    total_frames: Optional[int] = None,
    seed: int = 1,
    visualize_reward: bool = False,
    from_pixels: bool = True,
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_stack: int = 3,
    frame_skip: int = 2,
    episode_length: int = 1000,
    environment_kwargs: Optional[Dict] = None,
) -> gym.Env:
    """Build DeepMind Control Suite environments.

    Args:
        env_id (str): Name of environment.
        num_envs (int): Number of environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        resource_files (Optional[List]): File path of the resource files.
        img_source (Optional[str]): Type of the background distractor, supported values: ['color', 'noise', 'images', 'video'].
        total_frames (Optional[int]): for 'images' or 'video' distractor.
        seed (int): Random seed.
        visualize_reward (bool): True when 'from_pixels' is False, False when 'from_pixels' is True.
        from_pixels (bool): Provide image-based observations or not.
        height (int): Image observation height.
        width (int): Image observation width.
        camera_id (int): Camera id for generating image-based observations.
        frame_stack (int): Number of stacked frames.
        frame_skip (int): Number of action repeat.
        episode_length (int): Maximum length of an episode.
        environment_kwargs (Optional[Dict]): Other environment arguments.

    Returns:
        Environments instance.
    """

    def make_env(env_id: str, seed: int) -> Callable:
        def _thunk():
            domain_name, task_name = env_id.split("_")
            _env_id = f"dmc_{domain_name}_{task_name}_{seed}-v1"

            if from_pixels:
                assert not visualize_reward, "Cannot use visualize reward when learning from pixels!"

            # shorten episode length
            max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

            if _env_id not in registry.values():
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
                return gym.make(env_id)
            else:
                return FrameStack(gym.make(env_id), frame_stack)

        return _thunk

    envs = [make_env(env_id, seed + i) for i in range(num_envs)]
    envs = AsyncVectorEnv(envs)
    envs = RecordEpisodeStatistics(envs)

    return TorchVecEnvWrapper(envs, device)
