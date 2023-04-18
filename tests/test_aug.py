import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import cv2
import numpy as np
import torch

from hsuanwu.env.dmc import make_dmc_env
from hsuanwu.xplore.augmentation import RandomCrop

if __name__ == "__main__":
    env = make_dmc_env(
        env_id="hopper_hop",
        resource_files=None,
        img_source=None,
        total_frames=None,
        seed=1,
        visualize_reward=False,
        from_pixels=True,
        frame_skip=1,
        frame_stack=1,
    )

    obs, info = env.reset()

    obs_tensor = obs / 255.0
    print(obs_tensor.size())
    cv2.imwrite(
        "./tests/origin.jpg", np.transpose(obs_tensor.numpy()[0], [1, 2, 0]) * 255.0
    )

    aug = RandomCrop(pad=20, out=84)

    auged_obs = aug(obs_tensor)
    print(auged_obs.size())
    cv2.imwrite(
        "./tests/after.jpg", np.transpose(auged_obs.numpy()[0], [1, 2, 0]) * 255.0
    )
