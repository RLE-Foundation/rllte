from rllte.agent import PPO
from rllte.env import make_envpool_atari_env
from experimental.re3 import RE3

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    num_envs = 8
    env = make_envpool_atari_env(
        device=device,
        num_envs=num_envs,
    )
    
    # create agent and turn on pre-training mode
    agent = PPO(
        env=env, 
        device=device,
        tag="ppo_atari",
        discount=0.99,
        pretraining=True
    )
    
    # create intrinsic reward
    re3 = RE3(observation_space=env.observation_space,
              action_space=env.action_space,
              device=device)
    # set the reward module
    agent.set(reward=re3)
    # start training
    agent.train(num_train_steps=5000)