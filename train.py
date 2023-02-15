import jax.numpy as jnp
import time
import jax
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from hsuanwu.xploit.drqv2.agent import DrQv2Agent
from hsuanwu.xploit.replay_buffer import ReplayBuffer
from hsuanwu.envs.dmc import make_dmc_env, FrameStack

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
    print(env.observation_space, env.action_space)

    agent = DrQv2Agent(
        obs_shape = env.observation_space.shape, 
        action_shape = env.action_space.shape,
        feature_dim = 50,
        hidden_dim = 1024,
        lr = 1e-4,
        critic_target_tau = 0.01,
        num_expl_steps = 2000,
        update_every_steps = 2,
        stddev_schedule = 'linear(1.0, 0.1, 100000)',
        stddev_clip = 0.3)
    
    replay_buffer = ReplayBuffer(
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_envs=1)