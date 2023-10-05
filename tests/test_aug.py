import pytest
import torch as th

from rllte.env.testing import make_box_env
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
    envs = make_box_env(env_id="PixelObsEnv", num_envs=7, device=device, seed=1, asynchronous=True)

    obs, _ = envs.reset()
    aug = aug_cls().to(th.device(device))
    aug(obs / 255.0)

    print("Image augmentation test passed!")


@pytest.mark.parametrize("aug_cls", [RandomAmplitudeScaling, GaussianNoise])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_state_augmentation(aug_cls, device):
    envs = make_box_env(env_id="StateObsEnv", num_envs=7, device=device, seed=1, asynchronous=True)
    obs, _ = envs.reset()
    aug = aug_cls().to(th.device(device))
    aug(obs)

    print("State augmentation test passed!")
