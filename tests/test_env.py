import pytest
import torch as th

from rllte.env import (
    make_atari_env,
    make_bitflipping_env,
    make_bullet_env,
    make_dmc_env,
    make_envpool_atari_env,
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
        make_bullet_env,
        make_bitflipping_env,
        make_envpool_atari_env,
    ],
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("parallel", [True, False])
def test_discrete_env(env_cls, device, parallel):
    num_envs = 3
    if env_cls in [make_procgen_env, make_dmc_env]:
        env = env_cls(device=device, num_envs=num_envs)
    else:
        env = env_cls(device=device, num_envs=num_envs, parallel=parallel)
    time_step = env.reset()

    print(env.observation_space, env.action_space)

    for _ in range(10):
        action = env.action_space.sample()

        if env_cls in [make_atari_env, make_minigrid_env, make_procgen_env, make_envpool_atari_env, make_bitflipping_env]:
            action = th.randint(0, env.action_space.n, (num_envs,)).to(device)
        else:
            action = th.rand(size=(num_envs, env.action_space.shape[0])).to(device)

        time_step = env.step(action)
    env.close()

    print("Environment test passed!")
