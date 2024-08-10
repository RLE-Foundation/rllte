#!/bin/bash

# Define the list of intrinsic rewards
intrinsic_rewards=("extrinsic" "pseudocounts" "icm" "rnd" "ngu" "ride" "re3" "e3b" "disagreement")

# Loop over seeds (adjust the range as needed)
for seed in {1..3}; do
    # Loop over each intrinsic reward
    for reward in "${intrinsic_rewards[@]}"; do
        # Run with --obs_rms
        echo "Running with seed=${seed}, intrinsic_reward=${reward}, and obs_rms"
        python src/ppo.py --env_id=MiniHack-Room-5x5-v0 --intrinsic_reward=${reward} --seed=${seed} --obs_rms

        # Run without --obs_rms
        echo "Running with seed=${seed}, intrinsic_reward=${reward}, and without obs_rms"
        python src/ppo.py --env_id=MiniHack-Room-5x5-v0 --intrinsic_reward=${reward} --seed=${seed}
    done
done
