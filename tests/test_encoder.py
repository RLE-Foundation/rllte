import pytest
import torch as th

from rllte.env import make_dmc_env, make_minigrid_env
from rllte.xploit.encoder import (
    EspeholtResidualEncoder,
    IdentityEncoder,
    MnihCnnEncoder,
    PathakCnnEncoder,
    RaffinCombinedEncoder,
    TassaCnnEncoder,
    VanillaMlpEncoder,
)


@pytest.mark.parametrize(
    "encoder_cls",
    [
        IdentityEncoder,
        VanillaMlpEncoder,
        EspeholtResidualEncoder,
        TassaCnnEncoder,
        PathakCnnEncoder,
        MnihCnnEncoder,
        RaffinCombinedEncoder,
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_encoder(encoder_cls, device):
    num_envs = 3
    if encoder_cls in [IdentityEncoder, VanillaMlpEncoder]:
        envs = make_dmc_env(
            env_id="hopper_hop", seed=1, from_pixels=False, visualize_reward=True, device=device, num_envs=num_envs
        )
    elif encoder_cls in [RaffinCombinedEncoder]:
        envs = make_minigrid_env(num_envs=num_envs, device=device, fully_numerical=True, fully_observable=False)
    else:
        envs = make_dmc_env(
            env_id="hopper_hop", seed=1, from_pixels=True, visualize_reward=False, device=device, num_envs=num_envs
        )

    device = th.device(device)
    encoder = encoder_cls(observation_space=envs.observation_space, feature_dim=50).to(device)

    obs, _ = envs.reset()
    encoder(obs)

    print("Encoder test passed!")
