import numpy as np
import time
# from hsuanwu.xploit.drqv2.agent import DrQv2Agent
# from hsuanwu.xploit.replay_buffer import ReplayBuffer
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
    print(env.observation_space.shape, env.action_space.shape)
    print(env.observation_space.sample().shape)

    buffer = np.empty(shape=(1000000, )+ env.observation_space.shape, dtype=env.observation_space.dtype)
    env.reset()
    for i in range(1000000):
        obs, reward, done, info = env.step(env.action_space.sample())
        buffer[i] = obs
        print(i)

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
