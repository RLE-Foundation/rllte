import os
os.environ["OMP_NUM_THREADS"] = "1"
from rllte.xploit.agent import IMPALA
from rllte.env import make_atari_env
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_atari_env(
        env_id=args.env_id, 
        device=args.device,
        seed=args.seed, 
        num_envs=45, 
        distributed=True)
    
    eval_env = make_atari_env(
        env_id=args.env_id, 
        device=args.device,
        seed=args.seed,
        num_envs=1, 
        distributed=True)

    # create agent
    agent = IMPALA(
        env=env,
        eval_env=eval_env,
        tag=f"impala_atari_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        num_steps=80,
        num_actors=45,
        num_learners=4,
        num_storages=60,
        feature_dim=512,
    )
    # training
    agent.train(num_train_steps=30000000)