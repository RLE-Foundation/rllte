import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import cv2
import numpy as np
import torch

from hsuanwu.env.dmc import make_dmc_env
from hsuanwu.xplore.augmentation import (
    GrayScale,
    RandomColorJitter,
    RandomConvolution,
    RandomCrop,
    RandomCutout,
    RandomCutoutColor,
    RandomFlip,
    RandomRotate,
    RandomShift,
    RandomTranslate,
)

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
    )

    obs = env.reset()
    print("Obs Shape", obs.shape)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0) / 255.0
    print("Input Shape:", obs_tensor.shape)

    # Split the stack frames in Channels for cv2.imwrite stack-frame-1 -2 -3 ...
    frames_input = torch.chunk(obs_tensor, 3, dim=1)

    for i, frame in enumerate(frames_input):
        print(f"stack frame {i+1}: {frame.size()}")
        cv2.imwrite(
            "./tests/origin-stack-frame-" + str(i + 1) + ".png",
            np.transpose(frame.numpy()[0], [1, 2, 0]) * 255.0,
        )

    # Send obs_tensor to the Aug methods
    print("Augment Input Shape:", obs_tensor.size())

    # Test Some Methods in Xplore
    # aug =  RandomCutoutColor()
    # aug = RandomCutout()
    # aug = RandomColorJitter() Modify
    # aug = RandomConvolution()
    # aug = RandomCrop(pad=30, out=256)
    # aug = RandomShift(pad=30)
    # aug = RandomTranslate()
    # aug = RandomRotate()
    aug = GrayScale()

    # aug = RandomCrop(pad=20, out=84)

    # auged_obs = Crop(1).do_augmentation(obs_tensor)
    auged_obs = aug(obs_tensor)

    print("Output Shape:", auged_obs.size())
    frames_output = torch.chunk(auged_obs, 3, dim=1)

    for i, frame in enumerate(frames_output):
        print(f"stack frame {i+1}: {frame.size()}")
        cv2.imwrite(
            "./tests/after-stack-frame-" + str(i + 1) + ".png",
            np.transpose(frame.numpy()[0], [1, 2, 0]) * 255.0,
        )
