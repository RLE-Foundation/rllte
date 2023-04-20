The structure of Hsuanwu:
<div align=center>
<img src='/assets/images/structure.svg' style="width: 90%">
</div>

- **[Common](./common_index/index.md)**: Auxiliary modules like trainer and logger.
    + **Engine**: Engine for building Hsuanwu application.
    + **Logger**: Logger for managing output information.

- **[Xploit](./xploit_index/index.md)**: Modules that focus on <font color="#B80000"><b>exploitation</b></font> in RL.
    + **Encoder**: *Neural nework-based encoder for processing observations.*
    + **Learner**: *Agent for interacting and learning.*
    + **Storage**: *Buffer for storing collected experiences.*

- **[Xplore](./xplore_index/index.md)**: Modules that focus on <font color="#B80000"><b>exploration</b></font> in RL.
    + **Augmentation**: PyTorch.nn-like modules for observation augmentation.
    + **Distribution**: Distributions for sampling actions.
    + **Reward**: Intrinsic reward modules for enhancing exploration.

- **[Evaluation](./evaluation_index/index.md)**: Reasonable and reliable metrics for algorithm evaluation.

- **[Env](./env_index/index.md)**: Packaged environments (e.g., Atari games) for fast invocation.

- **[Pre-training](./pretraining_index/index.md)**: Methods of <font color="#B80000"><b>pre-training</b></font> in RL.

- **[Deployment](./deployment_index/index.md)**: Methods of <font color="#B80000"><b>model deployment</b></font> in RL.