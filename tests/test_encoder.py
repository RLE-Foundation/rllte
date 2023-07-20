import pytest
import torch as th

from rllte.env import make_dmc_env
from rllte.xploit.encoder import (
    EspeholtResidualEncoder,
    IdentityEncoder,
    MnihCnnEncoder,
    PathakCnnEncoder,
    TassaCnnEncoder,
    VanillaMlpEncoder,
    RaffinCombinedEncoder
)


@pytest.mark.parametrize(
    "encoder_cls",
    [IdentityEncoder, VanillaMlpEncoder, EspeholtResidualEncoder, TassaCnnEncoder, PathakCnnEncoder, MnihCnnEncoder, RaffinCombinedEncoder]
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_encoder(encoder_cls, device):
    if encoder_cls in [IdentityEncoder, VanillaMlpEncoder]:
        env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=False, visualize_reward=True, device=device)
    else:
        env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=True, visualize_reward=False, device=device)

    device = th.device(device)
    encoder = encoder_cls(observation_space=env.observation_space, feature_dim=50).to(device)

    obs = th.as_tensor(env.observation_space.sample(), device=device).unsqueeze(0)
    obs = encoder(obs)

    print("Encoder test passed!")
