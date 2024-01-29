from rllte.agent import PPO
from rllte.env import make_envpool_atari_env
from experimental.rnd import RND

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
    rnd = RND(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_envs=num_envs
    )

    # set the reward module
    agent.set(reward=rnd)
    # start training
    agent.train(num_train_steps=5000)