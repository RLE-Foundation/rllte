# update prop 0.1
python src/train_ppo.py --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=rms --seed=1
python src/train_ppo.py --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=rms --seed=2
python src/train_ppo.py --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=rms --seed=3