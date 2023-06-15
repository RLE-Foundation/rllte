import os
import sys
import pytest

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch as th
from rllte.env import make_dmc_env
from rllte.xploit.encoder import (IdentityEncoder,
                                  VanillaMlpEncoder,
                                  EspeholtResidualEncoder,
                                  TassaCnnEncoder,
                                  PathakCnnEncoder,
                                  MnihCnnEncoder)

@pytest.mark.parametrize("encoder", [IdentityEncoder, 
                                     VanillaMlpEncoder,
                                     EspeholtResidualEncoder,
                                     TassaCnnEncoder,
                                     PathakCnnEncoder,
                                     MnihCnnEncoder])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_encoder(encoder, device):
    if encoder in [IdentityEncoder, VanillaMlpEncoder]:
        env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=False, visualize_reward=True, device=device)
    else:
        env = make_dmc_env(env_id="hopper_hop", seed=1, from_pixels=True, visualize_reward=False, device=device)

    device = th.device(device)
    encoder = encoder(observation_space=env.observation_space, feature_dim=50).to(device)

    obs = th.as_tensor(env.observation_space.sample(), device=device).unsqueeze(0)
    obs = encoder(obs)

    print("Encoder test passed!")