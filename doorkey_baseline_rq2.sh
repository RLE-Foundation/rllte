# BASELINE 
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=pseudocounts --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=pseudocounts --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=pseudocounts --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ngu --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ngu --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ngu --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ride --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ride --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ride --seed=6

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --seed=4
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --seed=5
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --seed=6

# Q2 REW_MINMAX
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=icm --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=icm --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=icm --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=e3b --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=e3b --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=e3b --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=rnd --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=rnd --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=rnd --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=pseudocounts --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=pseudocounts --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=pseudocounts --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ngu --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ngu --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ngu --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=re3 --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=re3 --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=re3 --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ride --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ride --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ride --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=disagreement --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=disagreement --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=disagreement --seed=6


# Q2 NONE
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=icm --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=icm --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=icm --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=e3b --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=e3b --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=e3b --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=rnd --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=rnd --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=rnd --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=pseudocounts --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=pseudocounts --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=pseudocounts --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ngu --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ngu --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ngu --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=re3 --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=re3 --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=re3 --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ride --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ride --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ride --seed=6

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=disagreement --seed=4
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=disagreement --seed=5
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=disagreement --seed=6