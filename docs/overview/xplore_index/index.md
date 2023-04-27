+ **Reward**: Intrinsic reward modules for enhancing exploration.

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
|ProtoRL<sup>🐌</sup>| Entropy Maximization | ✔️ | ✔️ | [Reinforcement Learning with Prototypical Representations](http://proceedings.mlr.press/v139/yarats21a/yarats21a.pdf) |
|APS<sup>🐌</sup>| Skill Discovery | ✔️ | ✔️ | [APS: Active Pretraining with Successor Features](http://proceedings.mlr.press/v139/liu21b/liu21b.pdf) |

!!! tip "Tips of Reward"
    - **🐌**: Developing.
    - **Repr.**: The method involves representation learning.
    - **Visual**: The method works well in visual RL.

+ **Augmentation**: PyTorch.nn-like modules for observation augmentation.

|Module|Input|Reference|
|:-|:-|:-|
|GaussianNoise|States| [Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomAmplitudeScaling|States|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|AutoAugment|Images|[torchvision](https://pytorch.org/vision) |
|ElasticTransform|Images|[torchvision](https://pytorch.org/vision) |
|GrayScale|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomAdjustSharpness|Images| [torchvision](https://pytorch.org/vision) |
|RandomAugment|Images|[torchvision](https://pytorch.org/vision) |
|RandomAutocontrast|Images|[torchvision](https://pytorch.org/vision) |
|RandomColorJitter|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomConvolution|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomCrop|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomCutout|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomCutoutColor|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomEqualize|Images|[torchvision](https://pytorch.org/vision) |
|RandomFlip|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomInvert|Images|[torchvision](https://pytorch.org/vision) |[torchvision](https://pytorch.org/vision) |
|RandomPerspective|Images|[torchvision](https://pytorch.org/vision) |[torchvision](https://pytorch.org/vision) |
|RandomRotate|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |
|RandomShift|Images| [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)
|RandomTranslate|Images|[Reinforcement Learning with Augmented Data](https://proceedings.neurips.cc/paper/2020/file/e615c82aba461681ade82da2da38004a-Paper.pdf) |

+ **Distribution**: Distributions for sampling actions.

|Module|Type|Reference|
|:-|:-|:-|
|NormalNoise|Noise|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|OrnsteinUhlenbeckNoise|Noise|[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------)|
|TruncatedNormalNoise|Noise|[Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)|
|Categorical|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|DiagonalGaussian|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|
|SquashedNormal|Distribution|[torch.distributions](https://pytorch.org/docs/stable/distributions.html)|

!!! tip "Tips of Distribution"
  - In Hsuanwu, the action noise is implemented via a `Distribution` manner to realize unification.