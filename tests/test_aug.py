import pytest
import torch as th

from rllte.env.dmc import make_dmc_env
from rllte.xplore.augmentation import (
    GaussianNoise,
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
    RandomTranslate,
)


@pytest.mark.parametrize(
    "aug_cls",
    [
        GrayScale,
        Identity,
        RandomColorJitter,
        RandomConvolution,
        RandomCrop,
        RandomCutout,
        RandomCutoutColor,
        RandomFlip,
        RandomRotate,
        RandomShift,
        RandomTranslate,
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_image_augmentation(aug_cls, device):
    env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=True, visualize_reward=False, device=device)
    obs, _ = env.reset()
    aug = aug_cls().to(th.device(device))
    aug(obs / 255.0)

    print("Image augmentation test passed!")


@pytest.mark.parametrize("aug_cls", [RandomAmplitudeScaling, GaussianNoise])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_state_augmentation(aug_cls, device):
    env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=False, visualize_reward=True, device=device)
    obs, _ = env.reset()
    aug = aug_cls().to(th.device(device))
    aug(obs)

    print("State augmentation test passed!")
