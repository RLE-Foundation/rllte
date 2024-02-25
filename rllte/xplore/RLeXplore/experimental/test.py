import gymnasium as gym
env = gym.vector.SyncVectorEnv([
    lambda: gym.make("Pendulum-v1", g=9.81),
    lambda: gym.make("Pendulum-v1", g=1.62)
])

print(env.unwrapped.num_envs)
