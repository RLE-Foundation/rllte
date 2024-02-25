#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：ppo_re3_bullet.py
@Author ：YUAN Mingqi
@Date ：2022/9/19 21:29 
'''

import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import time
import torch
import argparse
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from rlexplore.re3 import RE3
from rlexplore.utils import create_env, cleanup_log_dir


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--action-space', type=str, default='cont')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--exploration', type=str, default='')
    parser.add_argument('--env-id', type=str, default='AssaultBullet-v0')
    parser.add_argument('--total-time-steps', type=int, default=10000000)
    parser.add_argument('--n-envs', type=int, default=10)
    parser.add_argument('--n-steps', type=int, default=128)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    log_dir = './logs/{}/{}/{}'.format(args.action_space, args.exploration, args.env_id)
    log_dir = os.path.expanduser(log_dir)
    cleanup_log_dir(log_dir)

    device = torch.device('cuda:0')

    num_episodes = int(args.total_time_steps / args.n_steps / args.n_envs)
    # Create vectorized environments.
    envs = create_env(env_id=args.env_id, n_envs=args.n_envs, log_dir=log_dir)
    # Create RE3 module.
    if args.exploration == 're3':
        re3 = RE3(obs_shape=envs.observation_space.shape, action_shape=envs.action_space.shape, device=device, latent_dim=128, beta=1e-2, kappa=1e-5)
    # Create PPO agent.
    model = PPO(policy='MlpPolicy', env=envs, n_steps=args.n_steps)
    # Set info buffer
    model.ep_info_buffer = deque(maxlen=10)
    _, callback = model._setup_learn(total_timesteps=args.total_time_steps, eval_env=None)

    t_s = time.perf_counter()
    all_eps_rewards = list()
    eps_rewards = deque([0.] * 10, maxlen=10)
    for i in range(num_episodes):
        model.collect_rollouts(
            env=envs,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=args.n_steps,
            callback=callback
        )
        # Compute intrinsic rewards.
        if args.exploration == 're3':
            intrinsic_rewards = re3.compute_irs(
                rollouts={'observations': model.rollout_buffer.observations},
                time_steps=i * args.n_steps * args.n_envs,
                k=3)
            model.rollout_buffer.rewards += intrinsic_rewards[:, :, 0]
        # Update policy using the currently gathered rollout buffer.
        model.train()
        t_e = time.perf_counter()

        eps_rewards.extend([ep_info["r"] for ep_info in model.ep_info_buffer])
        all_eps_rewards.append(list(eps_rewards.copy()))
        times_steps = i * args.n_steps * args.n_envs
        print('ENV {}, ALGO {}+{}, TOTAL TIME STEPS {}, FPS {} \n \
            MEAN|MEDIAN REWARDS {:.2f}|{:.2f}, MIN|MAX REWARDS {:.2f}|{:.2f}\n'.format(
            args.env_id, args.algo.upper(), args.exploration.upper(),
            times_steps, int(times_steps / (t_e - t_s)),
            np.mean(eps_rewards), np.median(eps_rewards), np.min(eps_rewards), np.max(eps_rewards)
        ))

    np.save(os.path.join(args.log_dir, 'episode_rewards.npy'), all_eps_rewards)
