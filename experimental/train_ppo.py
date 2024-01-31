import sys
sys.path.append("../")

from rllte.agent import PPO
from rllte.env import make_envpool_atari_env, make_mario_env
from rllte.xplore.reward.rnd import RND
from rllte.xplore.reward.icm import ICM
from rllte.xplore.reward.ride import RIDE
from rllte.xplore.reward.e3b import E3B
from rllte.xplore.reward.re3 import RE3
from rllte.xplore.reward.disagreement import Disagreement

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    num_envs = 32

    env = make_mario_env(
        device=device,
        num_envs=num_envs,
    )

    # create agent and turn on pre-training mode
    agent = PPO(
        env=env, 
        device=device,
        tag="ppo_mario",
        discount=0.99,
        batch_size=512,
        num_steps=1024,
        n_epochs=10,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        lr=2.5e-4,
        eps=1e-5,
        pretraining=True
    )
    
    # create intrinsic reward
    rnd = E3B(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_envs=num_envs,
        latent_dim=256
    )

    # set the reward module
    agent.set(reward=rnd)
    # start training
    agent.train(num_train_steps=20_000_000)