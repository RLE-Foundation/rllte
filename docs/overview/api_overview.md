- <font color="#009485"><b>Xploit</b></font>: Modules that focus on <font color="#B80000"><b>exploitation</b></font> in RL.
    + **Encoder**: *Neural nework-based encoder for processing observations.*
        - [VanillaCnnEncoder]()
        - [VanillaMlpEncoder]()

    + **Learner**: *Agent for interacting and learning.*
        - [ContinuousLearner]()
        - [DiscreteLearner]()

    + **Storage**: *Buffer for storing collected experiences.*
        - [VanillaReplayBuffer]()
        - [NStepReplayBuffer]()
        - [PrioritizedReplayBuffer]()
        - [VanillaRolloutBuffer]()

- <font color="#009485"><b>Xplore</b></font>: Modules that focus on <font color="#B80000"><b>exploration</b></font> in RL.
    + **Augmentation**: PyTorch.nn-like modules for observation augmentation.
        - [RandomCrop](../../api/xplore/augmentation/random_crop/)
        - [RandomFlip]()
        - [RandomShift](../../api/xplore/augmentation/random_shift/)

    + **Distribution**: Distributions for sampling actions.
        - [TruncatedNormal]()
        - [OrnsteinUhlenbeck]()

    + **Reward**: Intrinsic reward modules for enhancing exploration.
        - [ICM](../../api/xplore/reward/icm/)
        - [RND](../../api/xplore/reward/rnd/)
        - [GIRM](../../api/xplore/reward/girm/)
        - [NGU](../../api/xplore/reward/ngu/)
        - [RIDE](../../api/xplore/reward/ride/)
        - [RE3](../../api/xplore/reward/re3/)
        - [RISE](../../api/xplore/reward/rise/)
        - [REVD](../../api/xplore/reward/revd/)

- <font color="#009485"><b>Pre-training</b></font>: Methods of <font color="#B80000"><b>pre-training</b></font> in RL.

- <font color="#009485"><b>Deployment</b></font>: Methods of <font color="#B80000"><b>model deployment</b></font> in RL.