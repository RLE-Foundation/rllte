# BASELINE 
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=pseudocounts --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=pseudocounts --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=pseudocounts --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ngu --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ngu --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ngu --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ride --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ride --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=ride --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --seed=3

# Q2 REW_MINMAX
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=icm --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=icm --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=icm --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=e3b --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=e3b --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=e3b --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=rnd --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=rnd --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=rnd --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=pseudocounts --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=pseudocounts --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=pseudocounts --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ngu --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ngu --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ngu --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=re3 --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=re3 --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=re3 --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ride --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ride --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=ride --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=disagreement --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=disagreement --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=minmax --intrinsic_reward=disagreement --seed=3


# Q2 NONE
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=icm --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=icm --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=icm --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=e3b --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=e3b --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=e3b --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=rnd --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=rnd --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=rnd --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=pseudocounts --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=pseudocounts --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=pseudocounts --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ngu --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ngu --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ngu --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=re3 --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=re3 --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=re3 --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ride --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ride --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=ride --seed=3

sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=disagreement --seed=1
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=disagreement --seed=2
sbatch train_long  --env_id=MiniGrid-DoorKey-16x16-v0 --rwd_norm_type=none --intrinsic_reward=disagreement --seed=3