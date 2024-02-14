import argparse
from rllte.xplore.reward import *

def parse_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--env_id", type=str, default="SuperMarioBros-1-1-v3")
    parser.add_argument("--device", type=str, default="cuda:0")

    # train config
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--num_train_steps", type=int, default=10_000_000)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--clip_range_vf", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--anneal_lr", action="store_true", default=True)
    parser.add_argument("--init_fn", type=str, default="orthogonal")

    # ppo type
    parser.add_argument("--two_head", action="store_true", default=False)

    # intrinsic reward
    parser.add_argument("--intrinsic_reward", type=str, default="extrinsic")
    parser.add_argument("--rwd_norm_type", type=str, default="rms")
    parser.add_argument("--obs_rms", action="store_true", default=False)
    parser.add_argument("--update_proportion", type=float, default=1.0)
    parser.add_argument("--pretraining", action="store_true", default=False)
    parser.add_argument("--int_gamma", type=float, default=None)
    parser.add_argument("--weight_init", type=str, default="orthogonal")
    parser.add_argument("--parse_big", action="store_true", default=False)
    parser.add_argument("--beta", type=float, default=1.0)
    
    args = parser.parse_args()
    return args

def parse_args_big():
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
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--clip_range_vf", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--anneal_lr", action="store_true", default=False)
    parser.add_argument("--init_fn", type=str, default="orthogonal")

    # ppo type
    parser.add_argument("--two_head", action="store_true", default=False)

    # intrinsic reward
    parser.add_argument("--intrinsic_reward", type=str, default="extrinsic")
    parser.add_argument("--rwd_norm_type", type=str, default="rms")
    parser.add_argument("--obs_rms", action="store_true", default=False)
    parser.add_argument("--update_proportion", type=float, default=1.0)
    parser.add_argument("--pretraining", action="store_true", default=False)
    parser.add_argument("--int_gamma", type=float, default=None)
    parser.add_argument("--beta", type=float, default=1.0)

    parser.add_argument("--parse_big", action="store_true", default=False)
    
    args = parser.parse_args()
    return args

def make_env(args, device):
    # return either Mario or Atari env
    if "Mario" in args.env_id:
        from rllte.env import make_mario_env, make_mario_multilevel_env
        if "RandomStages" in args.env_id:
            env = make_mario_multilevel_env(
                device=device,
                num_envs=args.n_envs,
                env_id=args.env_id,
            )
        else:
            env = make_mario_env(
                device=device,
                num_envs=args.n_envs,
                env_id=args.env_id,
            )
    elif "MyWayHome" in args.env_id:
        from rllte.env import make_envpool_vizdoom_env
        env = make_envpool_vizdoom_env(
            device=device,
            num_envs=args.n_envs,
            env_id=args.env_id,
        )
    elif "procgen_" in args.env_id:
        from rllte.env import make_envpool_procgen_env
        args.env_id = args.env_id.split("_")[1] + "-v0"
        
        # this config gives a very cool level for MazeHard-v0
        env = make_envpool_procgen_env(
            env_id=args.env_id,
            num_envs=args.n_envs,
            device=device,
            num_levels=1,
            start_level=495,
            seed=394,
        )

    elif "MiniGrid" in args.env_id:
        from rllte.env import make_minigrid_env
        env = make_minigrid_env(
            env_id=args.env_id,
            num_envs=args.n_envs,
            device=device,
        )
    else:
        raise NotImplementedError

    return env, args.env_id

def select_intrinsic_reward(args, env, device):
    # chose encoder_model based on env_id. 
    if "Maze" in args.env_id:
        encoder_model = "espeholt"
    else:
        encoder_model = "mnih"
    
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
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "icm":
        intrinsic_reward = ICM(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "rnd":
        intrinsic_reward = RND(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "ngu":
        intrinsic_reward = NGU(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "ride":
        intrinsic_reward = RIDE(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "re3":
        intrinsic_reward = RE3(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "rise":
        intrinsic_reward = RISE(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
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
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "e3b":
        intrinsic_reward = E3B(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    elif args.intrinsic_reward == "disagreement":
        intrinsic_reward = Disagreement(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            n_envs=args.n_envs,
            rwd_norm_type=args.rwd_norm_type,
            obs_rms=args.obs_rms,
            update_proportion=args.update_proportion,
            gamma=args.int_gamma,
            encoder_model=encoder_model,
            weight_init=args.weight_init,
            beta=args.beta
        )
    else:
        raise NotImplementedError
    
    return intrinsic_reward