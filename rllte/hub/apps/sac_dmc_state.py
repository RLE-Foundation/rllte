import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
from rllte.xploit.agent import SAC
from rllte.xploit.storage import PrioritizedReplayStorage
from rllte.env import make_dmc_env
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="finger_spin")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_dmc_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        from_pixels=False,
        visualize_reward=True
    )
    eval_env = make_dmc_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        from_pixels=False,
        visualize_reward=True
    )
    # create agent
    agent = SAC(
        env=env,
        eval_env=eval_env,
        tag=f"sac_dmc_state_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        feature_dim=50,
        batch_size=1024,
        lr=0.0001,
        eps=1e-8,
        hidden_dim=1024,
        critic_target_tau=0.005,
        update_every_steps=2,
        network_init_method="orthogonal"
    )
    # storage = PrioritizedReplayStorage(
    #     observation_space=env.observation_space,
    #     action_space=env.action_space,
    #     device=args.device,
    #     storage_size=1000000,
    #     batch_size=1024
    # )
    # agent.set(storage=storage)
    # training
    agent.train(num_train_steps=250000)