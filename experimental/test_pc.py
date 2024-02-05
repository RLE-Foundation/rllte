import sys
sys.path.append("../")

from rllte.agent import PPO
from rllte.env import make_envpool_atari_env
from rllte.xplore.reward import *
from rllte.xploit.storage.roger_rollout_storage import RogerRolloutStorage

if __name__ == "__main__":
    # env setup
    device = "cuda:1"
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
        n_envs=num_envs,
        # episode_length=128,
        gamma=0.99,
    )

    # use the custom storage
    # please fill out the arguments by yourself
    storage = RogerRolloutStorage(
        ...
    )

# use the storage that don't cut the rewards when done is True
#==============================================================================
    # set the reward module
    agent.set(reward=rnd, storage=storage)
#==============================================================================
    # start training
    agent.train(num_train_steps=50000)
    