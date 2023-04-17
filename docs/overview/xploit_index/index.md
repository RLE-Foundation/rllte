+ **Encoder**: *Neural nework-based encoder for processing observations.*
    - [BaseEncoder](../../api/xploit/encoder/base)
    - [VanillaCnnEncoder](../../api/xploit/encoder/vanilla_cnn_encoder)
    - [VanillaMlpEncoder](../../api/xploit/encoder/vanilla_mlp_encoder)

+ **Learner**: *Agent for interacting and learning.*
    - [ContinuousLearner (DrQ-v2)](../../api/xploit/learner/drqv2)
    - [DiscreteLearner (PPG)](../../api/xploit/learner/ppg)

+ **Storage**: *Buffer for storing collected experiences.*
    - [VanillaReplayBuffer](../../api/xploit/storage/vanilla_replay_buffer)
    - [NStepReplayBuffer](../../api/xploit/storage/nstep_replay_buffer)
    - [PrioritizedReplayBuffer](../../api/xploit/storage/prioritized_replay_buffer)
    - [VanillaRolloutBuffer](../../api/xploit/storage/vanilla_rollout_buffer)