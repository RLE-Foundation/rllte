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
        domain_name="hopper",
        task_name="hop",
        resource_files=None,
        img_source=None,
        total_frames=None,
        seed=1,
        visualize_reward=False,
        from_pixels=True,
        frame_skip=1,
    )

    obs = env.reset()
    print(obs.shape)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0) / 255.0
    cv2.imwrite(
        "./tests/origin.png", np.transpose(obs_tensor.numpy()[0], [1, 2, 0]) * 255.0
    )
    print(obs_tensor.size())

    aug = RandomCrop(pad=20, out=84)

    # auged_obs = Crop(1).do_augmentation(obs_tensor)
    auged_obs = aug(obs_tensor)
    print(auged_obs.size())
    cv2.imwrite(
        "./tests/after.png", np.transpose(auged_obs.numpy()[0], [1, 2, 0]) * 255.0
    )
