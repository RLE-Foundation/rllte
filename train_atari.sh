# BASELINE 
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=disagreement --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=disagreement --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Seaquest-v5 --intrinsic_reward=disagreement --rwd_norm_type=minmax --seed=3

# Q1 OBS_RMS 
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=disagreement --seed=1
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=disagreement --seed=2
sbatch train_long --env_id=Gravitar-v5 --intrinsic_reward=disagreement --seed=3


# Private Eye
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=disagreement --seed=1
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=disagreement --seed=2
sbatch train_long --env_id=PrivateEye-v5 --intrinsic_reward=disagreement --seed=3

# Venture
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=Venture-v5 --intrinsic_reward=disagreement --seed=1
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=disagreement --seed=2
sbatch train_long --env_id=Venture-v5 --intrinsic_reward=disagreement --seed=3

# Venture
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=rnd --rwd_norm_type=none --obs_rms --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=pseudocounts --obs_rms --update_proportion=0.5 --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=ngu --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=ride --obs_rms --rwd_norm_type=minmax --seed=3

sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=disagreement --seed=1
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=disagreement --seed=2
sbatch train_long --env_id=MontezumaRevenge-v5 --intrinsic_reward=disagreement --seed=3