import pytest
import torch as th

from rllte.env import (
    make_atari_env,
    make_dmc_env,
    make_envpool_atari_env,
    make_envpool_procgen_env,
    make_minigrid_env,
    make_procgen_env,
)


@pytest.mark.parametrize(
    "env_cls",
    [
        make_atari_env,
        make_minigrid_env,
        make_procgen_env,
        make_dmc_env,
        make_envpool_atari_env,
        make_envpool_procgen_env
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_discrete_env(env_cls, device):
    num_envs = 3
    if env_cls in [make_procgen_env]:
        env = env_cls(device=device, num_envs=num_envs)
    else:
        # when set `asynchronous=True` for all the envs, 
        # the test will raise an EOF error
        env = env_cls(device=device, num_envs=num_envs, asynchronous=False)
    _ = env.reset()

    print(env.observation_space, env.action_space)

    for _ in range(10):
        action = env.action_space.sample()

        if env_cls in [make_atari_env, make_minigrid_env, make_procgen_env, 
                       make_envpool_atari_env, make_envpool_procgen_env]:
            action = th.randint(0, env.action_space.n, (num_envs,)).to(device)
        else:
            action = th.rand(size=(num_envs, env.action_space.shape[0])).to(device)

        _ = env.step(action)
    env.close()

    print("Environment test passed!")
