# update prop 0.1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=icm --rwd_norm_type=rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=icm --rwd_norm_type=rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=icm --rwd_norm_type=rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=rnd --rwd_norm_type=none --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=rnd --rwd_norm_type=none --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=rnd --rwd_norm_type=none --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.1 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=3

# update prop 0.5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=icm --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=icm --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=icm --rwd_norm_type=rms --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=rnd --rwd_norm_type=none --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=rnd --rwd_norm_type=none --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=rnd --rwd_norm_type=none --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=0.5 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=3


# update_prop 1.0
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=icm --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=icm --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=icm --rwd_norm_type=rms --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=e3b --rwd_norm_type=rms --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=rnd --rwd_norm_type=none --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=rnd --rwd_norm_type=none --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=rnd --rwd_norm_type=none --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=pseudocounts --rwd_norm_type=minmax --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ngu --rwd_norm_type=rms --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=re3 --rwd_norm_type=minmax --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=ride --rwd_norm_type=minmax --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --update_proportion=1.0 --intrinsic_reward=disagreement --rwd_norm_type=rms --seed=3