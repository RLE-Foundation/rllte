import argparse
from rllte.xplore.reward import *

def parse_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-1-1-v3")
    parser.add_argument("--device", type=str, default="cuda:0")

    # train config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--num_train_steps", type=int, default=10_000_000)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--clip_range_vf", type=float, default=None)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--init_fn", type=str, default="orthogonal")

    # intrinsic reward
    parser.add_argument("--intrinsic_reward", type=str, default="extrinsic")
    parser.add_argument("--rwd_norm_type", type=str, default="rms")
    parser.add_argument("--obs_rms", action="store_true", default=False)
    parser.add_argument("--update_proportion", type=float, default=1.0)
    args = parser.parse_args()
    return args

def make_env(args, device):
    # return either Mario or Atari env
    if "Mario" in args.env_id:
        from rllte.env import make_mario_env
        env = make_mario_env(
            device=device,
            num_envs=args.n_envs,
            env_id=args.env_id,
        )
        env_name = args.env_id
    else:
        try:
            from rllte.env import make_envpool_atari_env
            env = make_envpool_atari_env(
                env_id=args.env_id,
                num_envs=args.n_envs,
                device=device,
            )
            env_name = args.env_id
        except:
            raise NotImplementedError
    return env, env_name


def select_intrinsic_reward(args, env, device):
    # select intrinsic reward
    if args.intrinsic_reward == "extrinsic":
        intrinsic_reward = None
    elif args.intrinsic_reward == "pseudocounts":
        intrinsic_reward = PseudoCounts(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    elif args.intrinsic_reward == "icm":
        intrinsic_reward = ICM(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    elif args.intrinsic_reward == "rnd":
        intrinsic_reward = RND(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    elif args.intrinsic_reward == "ngu":
        intrinsic_reward = NGU(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    elif args.intrinsic_reward == "ride":
        intrinsic_reward = RIDE(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    elif args.intrinsic_reward == "re3":
        intrinsic_reward = RE3(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
        )
    elif args.intrinsic_reward == "rise":
        intrinsic_reward = RISE(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
        )
    elif args.intrinsic_reward == "revd":
        intrinsic_reward = REVD(
            observation_space=env.observation_space,
            action_space=env.action_space,
            episode_length=args.num_steps,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
        )
    elif args.intrinsic_reward == "e3b":
        intrinsic_reward = E3B(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    elif args.intrinsic_reward == "disagreement":
        intrinsic_reward = Disagreement(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion
        )
    else:
        raise NotImplementedError
    
    return intrinsic_reward