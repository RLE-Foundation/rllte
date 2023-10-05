import pytest
import torch as th

from rllte.env.testing import make_box_env
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
    num_envs = 7
    if encoder_cls in [IdentityEncoder, VanillaMlpEncoder]:
        envs = make_box_env(env_id="StateObsEnv", num_envs=num_envs, device=device, seed=1, asynchronous=True)
    elif encoder_cls in [RaffinCombinedEncoder]:
        envs = make_box_env(env_id="DictObsEnv", num_envs=num_envs, device=device, seed=1, asynchronous=True)
    else:
        envs = make_box_env(env_id="PixelObsEnv", num_envs=num_envs, device=device, seed=1, asynchronous=True)

    device = th.device(device)
    encoder = encoder_cls(observation_space=envs.observation_space, feature_dim=50).to(device)

    obs, _ = envs.reset()
    encoder(obs)

    print("Encoder test passed!")
