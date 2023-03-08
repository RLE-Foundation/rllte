import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path
import numpy as np
import time
import torch
# from hsuanwu.xploit.drqv2.agent import DrQv2Agent
# from hsuanwu.xploit.replay_buffer import ReplayBuffer
from hsuanwu.env.dmc import make_dmc_env, FrameStack
from hsuanwu.xploit.storage.nstep_replay_buffer import NStepReplayBuffer

if __name__ == '__main__':
    env = make_dmc_env(domain_name='hopper', 
                       task_name='hop', 
                       resource_files=None, 
                       img_source=None,
                       total_frames=None,
                       seed=1, 
                       visualize_reward=False, 
                       from_pixels=True, 
                       frame_skip=1)
    env = FrameStack(env, k=2)
    print(env.observation_space.shape, env.action_space.shape)
    print(env.observation_space.sample().shape)

    replay_buffer = NStepReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        buffer_size=1000000
    )

    loader = torch.utils.data.DataLoader(replay_buffer,
                                         batch_size=256,
                                         num_workers=1,
                                         pin_memory=True)

    obs = env.reset()
    for step in range(1000000):
        print(step)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        replay_buffer.add(
            obs,
            action,
            reward,
            done,
            discount=info['discount']
        )

        if step > 1000:
            batch = next(iter(loader)) #replay_buffer.sample(batch_size=256)
            obs, action, reward, discount, next_obs = batch
            print(obs.shape, action.shape, reward.shape, discount.shape)
            quit(0)

        if done:
            obs = env.reset()

    # agent = DrQv2Agent(
    #     obs_shape = env.observation_space.shape, 
    #     action_shape = env.action_space.shape,
    #     feature_dim = 50,
    #     hidden_dim = 1024,
    #     lr = 1e-4,
    #     critic_target_tau = 0.01,
    #     num_expl_steps = 50,
    #     update_every_steps = 2,
    #     stddev_schedule = 'linear(1.0, 0.1, 100000)',
    #     stddev_clip = 0.3)
    
    # replay_buffer = ReplayBuffer(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     buffer_size=2000,
    #     n_envs=1)
    
    # obs = env.reset()
    # replay_buffer.observations[0] = obs
    
    # s = time.perf_counter()
    # for step in range(1500):
    #     action = agent.act(obs=replay_buffer.observations[step])
    #     action = np.asarray(action[0])
    #     obs, reward, done, info = env.step(action)
    #     # print(step, reward, done, info['discount'])
    #     if not done or 'TimeLimit.truncated' in info:
    #         mask = 1.0
    #     else:
    #         mask = 0.0
    #     replay_buffer.add(obs=obs, action=action, reward=reward, mask=mask, done=done)
    # e = time.perf_counter()
    # print(e - s)
