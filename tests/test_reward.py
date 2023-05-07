import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.env import make_atari_env, make_dmc_env
from hsuanwu.xplore.reward import RE3, ICM
import torch as th


if __name__ == '__main__':
    train_env = make_atari_env(
    env_id='Alien-v5',
    num_envs=8,
    seed=1,
    frame_stack=4,
    device='cuda')

    print(train_env.observation_space, train_env.action_space)
    irs = ICM(
        obs_space=train_env.observation_space,
        action_space=train_env.action_space,
        device='cuda',
    )

    # for testing ride
    obs1 = th.ones((1, 4, 84, 84))
    obs2 = th.ones((1, 4, 84, 84)) * 3
    obs3 = th.ones((1, 4, 84, 84))
    obs4 = th.ones((1, 4, 84, 84)) * 5.5
    obs = th.stack([obs1, obs2, obs3, obs4], dim=0)
    print(obs.size())

    samples = {
        'obs': obs[:-1],
        'actions': th.randint(low=0, high=3, size=(3, 1, 1)),
        'next_obs': obs[1:],
    }

    for i in range(10):
        rewards = irs.compute_irs(samples=samples, step=0)
        print(rewards.cpu().numpy().tolist(), rewards.device)
    
    train_env = make_dmc_env(env_id='cartpole_swingup', 
                         num_envs=1,
                         seed=1, 
                         visualize_reward=True,
                         from_pixels=False
                         )
    print(train_env.observation_space, train_env.action_space)
    irs = RE3(
        obs_space=train_env.observation_space,
        action_space=train_env.action_space,
        device='cuda',
        average_entropy=True
    )
    samples = {
        'obs': th.rand(size=(256, 1, 5)),
        'actions': th.rand(size=(256, 1, 1)),
        'next_obs': th.rand(size=(256, 1, 5))
    }

    for i in range(10):
        rewards = irs.compute_irs(samples=samples, step=0)
        print(rewards.cpu().numpy().tolist(), rewards.device)
