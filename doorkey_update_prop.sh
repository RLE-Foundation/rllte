# update prop 0.1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=icm --rwd_norm_type=rms --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=icm --rwd_norm_type=rms --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=icm --rwd_norm_type=rms --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=rnd --rwd_norm_type=none --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=rnd --rwd_norm_type=none --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=rnd --rwd_norm_type=none --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=6

# update prop 0.5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=icm --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=icm --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=icm --rwd_norm_type=rms --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=rnd --rwd_norm_type=none --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=rnd --rwd_norm_type=none --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=rnd --rwd_norm_type=none --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=6


# update_prop 1.0
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=icm --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=icm --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=icm --rwd_norm_type=rms --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=rnd --rwd_norm_type=none --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=rnd --rwd_norm_type=none --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=rnd --rwd_norm_type=none --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=6