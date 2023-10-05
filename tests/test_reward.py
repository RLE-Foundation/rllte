import pytest
import torch as th

from rllte.env.testing import (make_box_env, 
                               make_discrete_env,
                               make_multibinary_env,
                               make_multidiscrete_env
                               )
from rllte.xplore.reward import GIRM, ICM, NGU, RE3, REVD, RIDE, RISE, RND, PseudoCounts


@pytest.mark.parametrize("reward", [GIRM, ICM, NGU, PseudoCounts, RE3, RIDE, RISE, RND, REVD])
@pytest.mark.parametrize("env_cls", [make_box_env, make_discrete_env, make_multidiscrete_env, make_multibinary_env])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_reward(reward, env_cls, device):
    num_envs = 3
    num_steps = 256

    env = env_cls(device=device, num_envs=num_envs)
    device = th.device(device)
    env.reset()

    if reward in [RE3, RISE]:
        irs = reward(observation_space=env.observation_space, action_space=env.action_space, device=device, num_envs=num_envs)
    else:
        irs = reward(observation_space=env.observation_space, action_space=env.action_space, device=device)

    obs = th.rand(size=(num_steps, num_envs, *env.observation_space.shape)).to(device)
    if env_cls is make_discrete_env:
        action = th.randint(0, env.action_space.n, (num_steps, num_envs)).to(device)
    elif env_cls is make_box_env:
        action = th.rand(size=(num_steps, num_envs, env.action_space.shape[0])).to(device)
    elif env_cls is make_multidiscrete_env:
        action = th.randint(0, 2, (num_steps, num_envs, env.action_space.nvec.shape[0])).float().to(device)
    elif env_cls is make_multibinary_env:
        action = th.randint(0, 2, (num_steps, num_envs, env.action_space.n)).float().to(device)

    for i in range(num_steps):
        irs.add(
            samples={
                'obs': obs[i],
                'actions': action[i],
                'next_obs': obs[i]
            }
        )

    samples = {
        "obs": obs,
        "actions": action,
        "next_obs": obs,
    }

    irs.compute_irs(samples)

    print("Intrinsic reward test passed!")
