import os
import sys
import pytest

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch as th

from rllte.env.dmc import make_dmc_env
from rllte.xplore.augmentation import (GaussianNoise,
                                       GrayScale,
                                       Identity,
                                       RandomAmplitudeScaling,
                                       RandomColorJitter,
                                       RandomConvolution,
                                       RandomCrop,
                                       RandomCutout,
                                       RandomCutoutColor,
                                       RandomFlip,
                                       RandomRotate,
                                       RandomShift,
                                       RandomTranslate)

@pytest.mark.parametrize("aug", [GrayScale, 
                                 Identity,
                                 RandomColorJitter,
                                 RandomConvolution,
                                 RandomCrop,
                                 RandomCutout,
                                 RandomCutoutColor,
                                 RandomFlip,
                                 RandomRotate,
                                 RandomShift,
                                 RandomTranslate])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_image_augmentation(aug, device):
    env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=True, visualize_reward=False, device=device)
    obs, info = env.reset()
    aug = aug().to(th.device(device))
    auged_obs = aug(obs / 255.0)

    print("Image augmentation test passed!")

@pytest.mark.parametrize("aug", [RandomAmplitudeScaling, GaussianNoise])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_state_augmentation(aug, device):
    env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=False, visualize_reward=True, device=device)
    obs, info = env.reset()
    aug = aug().to(th.device(device))
    auged_obs = aug(obs)

    print("State augmentation test passed!")