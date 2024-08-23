# BASELINE 
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --obs_rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --obs_rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=extrinsic --obs_rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --obs_rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --obs_rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=icm --obs_rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --obs_rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --obs_rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=e3b --obs_rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --obs_rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --obs_rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=rnd --obs_rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --obs_rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --obs_rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=re3 --obs_rms --seed=3

sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --obs_rms --seed=1
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --obs_rms --seed=2
sbatch train_long --env_id=MiniGrid-DoorKey-16x16-v0 --intrinsic_reward=disagreement --obs_rms --seed=3