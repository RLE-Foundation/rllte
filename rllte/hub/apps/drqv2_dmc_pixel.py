from rllte.xploit.agent import DrQv2
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
        from_pixels=True,
        visualize_reward=False,
        frame_stack=3,
        frame_skip=2,
    )
    eval_env = make_dmc_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        from_pixels=True,
        visualize_reward=False,
        frame_stack=3,
        frame_skip=2,
    )
    # create agent
    agent = DrQv2(
        env=env,
        eval_env=eval_env,
        tag=f"drqv2_dmc_pixel_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        feature_dim=50,
        batch_size=256,
        lr=0.0001,
        eps=1e-8,
        hidden_dim=1024,
        critic_target_tau=0.01,
        update_every_steps=2,
        network_init_method="orthogonal"
    )
    # training
    agent.train(num_train_steps=250000)