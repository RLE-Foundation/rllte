from src.utils import parse_args, make_env, select_intrinsic_reward
from rllte.agent import PPO

if __name__ == "__main__":
    # env setup
    args = parse_args()

    env, env_name = make_env(args, args.device)
    
    # select intrinsic reward
    intrinsic_reward = select_intrinsic_reward(
        args=args,
        env=env,
        device=args.device,
    )
    
    exp_name = f"PPO_{env_name}_{args.intrinsic_reward}_obsRMS:{args.obs_rms}_rewNorm:{args.rwd_norm_type}_updateProp:{args.update_proportion}_weightInit:{args.weight_init}_s{args.seed}"
   
    ppo_args = dict(
        env=env, 
        seed=args.seed,
        device=args.device,
        tag=exp_name,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eps=args.eps,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        discount=args.discount,
        init_fn=args.init_fn,
        pretraining=args.pretraining,
    )

    agent = PPO(**ppo_args)
        
    # create intrinsic reward
    if intrinsic_reward is not None:
        agent.set(reward=intrinsic_reward)
        
    print("==== AGENT ====")
    print(agent.encoder)
    print(agent.policy)

    # start training
    agent.train(
        num_train_steps=args.num_train_steps,
        anneal_lr=args.anneal_lr,
        save_interval=999_999_999,
        th_compile=False,
    )