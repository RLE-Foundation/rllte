<!-- <div align=center>
<img src='/assets/images/structure.svg' style="width: 100%">
</div> -->

#### <font color="#B80000"><b>Agent</b></font>: Implemented RL algorithms using **RLLTE** modules.

| Type 	| Legacy 	| Current 	|
|---	|---	|---	|
| On-Policy 	| [A2C](https://arxiv.org/abs/1602.01783)<sup>🖥️⛓️💰</sup>,[PPO](https://arxiv.org/pdf/1707.06347)<sup>🖥️⛓️💰</sup> 	| [DAAC](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf)<sup>🖥️⛓️💰</sup>,[DrAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)<sup>🖥️⛓️💰🔭</sup>,[DrDAAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)<sup>🖥️⛓️💰🔭</sup> 	|
| Off-Policy 	| [DQN](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)<sup>🖥️⛓️💰</sup>,[DDPG](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------)<sup>🖥️⛓️💰</sup>,[SAC](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)<sup>🖥️⛓️💰</sup> 	| [DrQ-v2](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)<sup>🖥️⛓️💰🔭</sup> 	|
| Distributed 	|  	| [IMPALA](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)<sup>⛓️</sup> 	|

> - 🖥️: Support Neural-network processing unit.
> - ⛓️: Multi Processing.
> - 💰: Support intrinsic reward shaping.
> - 🔭: Support observation augmentation.

<!-- #### <font color="#0053D6"><b>Common</b></font>: Base classes and auxiliary modules. -->

#### <font color="#B80000"><b>Xploit</b></font>: Modules that focus on <font color="#B80000"><b>exploitation</b></font> in RL.


!!! abstract "Policy: *Policies for interaction and learning.*"
    - OnPolicySharedActorCritic
    - OnPolicyDecoupledActorCritic
    - OffPolicyDoubleQNetwork
    - OffPolicyDeterministicActorDoubleCritic
    - OffPolicyStochasticActorDoubleCritic
    - DistributedActorLearner

!!! abstract "Encoder: *Neural nework-based encoders for processing observations.*"
    **Naming Rule**: `Surname of the first author` + `Backbone` + `Encoder`

    - EspeholtResidualEncoder
    - IdentityEncoder
    - MnihCnnEncoder
    - TassaCnnEncoder
    - PathakCnnEncoder
    - VanillaMlpEncoder
    - RaffinCombinedEncoder



!!! abstract "Storage: *Storges for storing collected experiences.*"
    - VanillaRolloutStorage (On-policy)
    - DictRolloutStorage (On-policy)
    - VanillaReplayStorage (Off-policy)
    - DictReplayStorage (Off-policy)
    - NStepReplayStorage (Off-policy)
    - PrioritizedReplayStorage (Off-policy)
    - HerReplayStorage (Off-policy)
    - VanillaDistributedStorage (Distributed)

#### <font color="#B80000"><b>Xplore</b></font>: Modules that focus on <font color="#B80000"><b>exploration</b></font> in RL.
!!! abstract "Augmentation: *PyTorch.nn-like modules for observation augmentation.*"
    - GaussianNoise (States)
    - RandomAmplitudeScaling (States)
    - GrayScale (Images)
    - RandomColorJitter (Images)
    - RandomConvolution (Images)
    - RandomCrop (Images)
    - RandomCutout (Images)
    - RandomCutoutColor (Images)
    - RandomFlip (Images)
    - RandomRotate (Images)
    - RandomShift (Images)
    - RandomTranslate (Images)

!!! abstract "Distribution: *Distributions for sampling actions.*"
    In **RLLTE**, the action noise is implemented via a `Distribution` manner to realize unification.

    - NormalNoise (Noise)
    - OrnsteinUhlenbeckNoise (Noise)
    - TruncatedNormalNoise (Noise)
    - Bernoulli (Distribution)
    - Categorical (Distribution)
    - DiagonalGaussian (Distribution)
    - SquashedNormal (Distribution)
    - MultiCategorical (Distribution)

!!! abstract "Reward: *Intrinsic reward modules for enhancing exploration.*"

    | **Type** 	| **Modules** 	|
    |---	|---	|
    | Count-based 	| [PseudoCounts](https://arxiv.org/pdf/2002.06038), [RND](https://arxiv.org/pdf/1810.12894.pdf) 	|
    | Curiosity-driven 	| [ICM](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf), [GIRM](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf), [RIDE](https://arxiv.org/pdf/2002.12292) 	|
    | Memory-based 	| [NGU](https://arxiv.org/pdf/2002.06038) 	|
    | Information theory-based 	| [RE3](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf), [RISE](https://ieeexplore.ieee.org/abstract/document/9802917/), [REVD](https://openreview.net/pdf?id=V2pw1VYMrDo) 	|

    See [Tutorials: Use Intrinsic Reward and Observation Augmentation](./tutorials/data_augmentation.md) for usage examples.

#### <font color="#B80000"><b>Env</b></font>: Packaged environments (e.g., Atari games) for fast invocation.

|Function|Name|Remark|Reference|
|:-|:-|:-|:-|
|make_atari_env|Atari Games|Discrete control|[Paper](https://www.jair.org/index.php/jair/article/download/10819/25823)|
|make_bullet_env|PyBullet Robotics Environments|Continuous control|[Paper](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)|
|make_dmc_env|DeepMind Control Suite|Continuous control|[Paper](https://arxiv.org/pdf/1801.00690)|
|make_minigrid_env|MiniGrid Games|Discrete control|[Paper](https://github.com/Farama-Foundation/Minigrid)|
|make_procgen_env|Procgen Games|Discrete control|[Paper](http://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf)|
|make_robosuite_env|Robosuite Robotics Environments|Continuous control|[Paper](http://robosuite.ai/)|

#### <font color="#B80000"><b>Copilot</b></font>: Large language model-empowered copilot.
See [Copilot](./copilot.md).

#### <font color="#B80000"><b>Hub</b></font>: Fast training API and reusable benchmarks.
See [Benchmarks](./benchmarks.md).

#### <font color="#B80000"><b>Evaluation</b></font>: Reasonable and reliable metrics for algorithm evaluation.
See [Tutorials: Evaluate Your Model](./tutorials/evaluation.md).

#### <font color="#B80000"><b>Pre-training</b></font>: Methods of pre-training in RL.

See [Tutorials: Pre-training](./tutorials/pre-training.md).

#### <font color="#B80000"><b>Deployment</b></font>: Methods of model deployment in RL.

See [Tutorials: Deploy Your Model in Inference Devices](./tutorials/deployment.md).