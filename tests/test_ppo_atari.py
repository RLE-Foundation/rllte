import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path
from hsuanwu.common.logger import Logger

# env part
from hsuanwu.env import make_atari_env
from hsuanwu.xploit.encoder import ResNetEncoder, VanillaCnnEncoder
from hsuanwu.xploit.storage import VanillaRolloutBuffer
from hsuanwu.xploit.learner import PPOAgent
from hsuanwu.xplore.distribution import Categorical

import hydra
import torch
import time
import numpy as np
torch.backends.cudnn.benchmark = True

num_steps = 256
num_envs = 2

device = torch.device('cuda')
train_env = make_atari_env(env_id='Assault-v5', num_envs=num_envs, seed=0, frame_stack=4, device=device)
test_env = make_atari_env(env_id='Assault-v5', num_envs=num_envs, seed=0, frame_stack=4, device=device)


encoder = ResNetEncoder(observation_space=train_env.observation_space, feature_dim=256).to(device)

learner = PPOAgent(
    observation_space=train_env.observation_space,
    action_space=train_env.single_action_space,
    action_type='dis',
    device='cuda',
    feature_dim=256,
    hidden_dim=256,
    lr=2.5e-4
)
learner.set_encoder(encoder)
learner.set_dist(Categorical)

rollout_buffer = VanillaRolloutBuffer(device='cuda',
                                      obs_shape=train_env.observation_space.shape,
                                      action_shape=train_env.single_action_space.n,
                                      action_type='dis',
                                      num_steps=num_steps,
                                      num_envs=num_envs)

obs = train_env.reset()
num_updates = 10000000 // num_envs // num_steps

for update in range(10):
    for step in range(num_steps):
        with torch.no_grad():
            actions, values, log_probs, entropy = learner.act(obs, training=True, step=0)
        next_obs, rewards, dones, infos = train_env.step(actions)
        # print(actions.shape, values.shape, rewards.shape, log_probs.size(), entropy.size())

        rollout_buffer.add(obs=obs, 
                           actions=actions, 
                           rewards=rewards, 
                           dones=dones, 
                           log_probs=log_probs, 
                           values=values)
    
        obs = next_obs
    
    # get the value estimation of the last step
    with torch.no_grad():
        last_values = learner.get_value(next_obs).detach()
    
    # perform return and advantage estimation
    rollout_buffer.compute_returns_and_advantages(last_values)

    # policy update
    train_metrics = learner.update(rollout_buffer)
    print(train_metrics)

    # reset buffer
    rollout_buffer.reset()