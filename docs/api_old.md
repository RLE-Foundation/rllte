<div align=center>
<img src='/assets/images/structure.svg' style="width: 100%">
</div>

### <font color="#0053D6"><b>Common</b></font>: Auxiliary modules like trainer and logger.
- **Engine**: *Engine for building Hsuanwu application.*
- **Logger**: *Logger for managing output information.*

### <font color="#0053D6"><b>Xploit</b></font>: Modules that focus on <font color="#B80000"><b>exploitation</b></font> in RL.
+ **Agent**: *Agent for interacting and learning.*

| Type 	| Algorithm 	|
|---	|---	|
| On-Policy 	| [A2C](https://arxiv.org/abs/1602.01783)<sup>🖥️⛓️💰</sup>,[PPO](https://arxiv.org/pdf/1707.06347)<sup>🖥️⛓️💰</sup> 	[DAAC](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf)<sup>🖥️⛓️💰</sup>,[DrAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)<sup>🖥️⛓️💰🔭</sup>,[DrDAAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)<sup>🖥️⛓️💰🔭</sup> 	|
| Off-Policy 	| [DQN](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)<sup>🖥️⛓️💰</sup>,[DDPG](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------)<sup>🖥️⛓️💰</sup>,[SAC](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)<sup>🖥️⛓️💰</sup> 	[DrQ-v2](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)<sup>🖥️⛓️💰🔭</sup> 	|
| Distributed 	|   [IMPALA](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)<sup>⛓️</sup> 	|

> - 🖥️: Support Neural-network processing unit.
> - ⛓️: Multi Processing.
> - 💰: Support intrinsic reward shaping.
> - 🔭: Support observation augmentation.

| Module | Recurrent | Box | Discrete | MultiBinary | Multi Processing | NPU | Paper | Citations |
|---|---|---|---|---|---|---|---|---|
| SAC | ❌ | ✔️ | ❌ | ❌ | ❌ | ✔️ | [Link](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) | 5077⭐ |
| DrQ | ❌ | ✔️ | ❌ | ❌ | ❌ | ✔️ | [Link](https://arxiv.org/pdf/2004.13649) | 433⭐ |
| DDPG | ❌ | ✔️ | ❌ | ❌ | ❌ | ✔️ | [Link](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------) | 11819⭐ |
| DrQ-v2 | ❌ | ✔️ | ❌ | ❌ | ❌ | ✔️ | [Link](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com) | 100⭐ |
| DAAC | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | [Link](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf) | 56⭐ |
| PPO | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | [Link](https://arxiv.org/pdf/1707.06347) | 11155⭐ |
| DrAC | ❌ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | [Link](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf) | 29⭐ |
| IMPALA | ✔️ | ✔️ | ✔️ | ❌ | ✔️ | ✔️ | [Link](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) | 1219⭐ |


!!! tip "Tips of Agent"
    - 🐌: Developing.
    - **NPU**: Support Neural-network processing unit.
    - **Recurrent**: Support recurrent neural network.
    - **Box**: A N-dimensional box that containes every point in the action space.
    - **Discrete**: A list of possible actions, where each timestep only one of the actions can be used.
    - **MultiBinary**: A list of possible actions, where each timestep any of the actions can be used in any combination.

+ **Encoder**: *Neural nework-based encoder for processing observations.*

|Module|Input|Reference|Target Task|
|:-|:-|:-|:-|
|EspeholtResidualEncoder|Images|[IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)|Atari or Procgen games.|
|IdentityEncoder|States|N/A|DeepMind Control Suite: state|
|MnihCnnEncoder|Images|[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf?source=post_page---------------------------)|Atari games.|
|TassaCnnEncoder|Images|[DeepMind Control Suite](https://arxiv.org/pdf/1801.00690)|DeepMind Control Suite: pixel|
|PathakCnnEncoder|Images|[Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)|Atari or MiniGrid games|
|VanillaMlpEncoder|States|N/A|DeepMind Control Suite: state|

!!! tip "Tips of Encoder"
    - **Naming Rule**: 'Surname of the first author' + 'Backbone' + 'Encoder'
    - **Input**: Input type.
    - **Target Task**: The testing tasks in their paper or potential tasks.

+ **Storage**: *Storge for storing collected experiences.*

|Module|Remark|
|:-|:-|
|VanillaRolloutStorage|On-Policy RL|
|VanillaReplayStorage|Off-Policy RL|
|NStepReplayStorage|Off-Policy RL|
|PrioritizedReplayStorage|Off-Policy RL|
|DistributedStorage|Distributed RL|

### <font color="#0053D6"><b>Xplore</b></font>: Modules that focus on <font color="#B80000"><b>exploration</b></font> in RL.
+ **Augmentation**: *PyTorch.nn-like modules for observation augmentation.*

|Module|Input|Reference|
|:-|:-|:-|
|GaussianNoise|States| [Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomAmplitudeScaling|States|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|GrayScale|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomColorJitter|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomConvolution|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomCrop|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomCutout|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomCutoutColor|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomFlip|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomRotate|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomShift|Images| [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)
|RandomTranslate|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |

+ **Distribution**: *Distributions for sampling actions.*

|Module|Type|Reference|
|:-|:-|:-|
|NormalNoise|Noise|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|OrnsteinUhlenbeckNoise|Noise|[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------)|
|TruncatedNormalNoise|Noise|[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)|
|Bernoulli|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|Categorical|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|DiagonalGaussian|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|SquashedNormal|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|

!!! tip "Tips of Distribution"
    - In Hsuanwu, the action noise is implemented via a `Distribution` manner to realize unification.

+ **Reward**: *Intrinsic reward modules for enhancing exploration.*

| Module | Remark | Repr.  | Visual | Reference | 
|:-|:-|:-|:-|:-|
| PseudoCounts | Count-Based exploration |✔️|✔️|[Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038) |
| ICM  | Curiosity-driven exploration  | ✔️|✔️| [Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf) | 
| RND  | Count-based exploration  | ❌|✔️| [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf) | 
| GIRM | Curiosity-driven exploration  | ✔️ |✔️| [Intrinsic Reward Driven Imitation Learning via Generative Model](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf)|
| NGU | Memory-based exploration  | ✔️  |✔️| [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038) | 
| RIDE| Procedurally-generated environment | ✔️ |✔️| [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://arxiv.org/pdf/2002.12292)|
| RE3  | Entropy Maximization | ❌ |✔️| [State Entropy Maximization with Random Encoders for Efficient Exploration](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf) |
| RISE  | Entropy Maximization  | ❌  |✔️| [Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9802917/) | 
| REVD  | Divergence Maximization | ❌  |✔️| [Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning](https://openreview.net/pdf?id=V2pw1VYMrDo)|

!!! tip "Tips of Reward"
    - **🐌**: Developing.
    - **Repr.**: The method involves representation learning.
    - **Visual**: The method works well in visual RL.

See [Tutorials: Use intrinsic reward and observation augmentation](./tutorials/data_augmentation.md) for usage examples.

### <font color="#0053D6"><b>Evaluation</b></font>: Reasonable and reliable metrics for algorithm <font color="#B80000"><b>evaluation</b></font>.
See [Tutorials: Evaluate your model](./tutorials/evaluation.md).

### <font color="#0053D6"><b>Env</b></font>: Packaged <font color="#B80000"><b>environments</b></font> (e.g., Atari games) for fast invocation.

|Module|Name|Remark|Reference|
|:-|:-|:-|:-|
|make_atari_env|Atari Games|Discrete control|[The Arcade Learning Environment: An Evaluation Platform for General Agents](https://www.jair.org/index.php/jair/article/download/10819/25823)|
|make_bullet_env|PyBullet Robotics Environments|Continuous control|[Pybullet: A Python Module for Physics Simulation for Games, Robotics and Machine Learning](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)|
|make_dmc_env|DeepMind Control Suite|Continuous control|[DeepMind Control Suite](https://arxiv.org/pdf/1801.00690)|
|make_minigrid_env|MiniGrid Games|Discrete control|[Minimalistic Gridworld Environment for Gymnasium](https://github.com/Farama-Foundation/Minigrid)|
|make_procgen_env|Procgen Games|Discrete control|[Leveraging Procedural Generation to Benchmark Reinforcement Learning](http://proceedings.mlr.press/v119/cobbe20a/cobbe20a.pdf)|
|make_robosuite_env|Robosuite Robotics Environments|Continuous control|[Robosuite: A Modular Simulation Framework and Benchmark for Robot Learning](http://robosuite.ai/)|

### <font color="#0053D6"><b>Pre-training</b></font>: Methods of <font color="#B80000"><b>pre-training</b></font> in RL.

See [Tutorials: Pre-training in Hsuanwu](./tutorials/pre-training.md).

### <font color="#0053D6"><b>Deployment</b></font>: Methods of model <font color="#B80000"><b>deployment</b></font> in RL.

See [Tutorials: Deploy your model in inference devices](./tutorials/deployment.md).