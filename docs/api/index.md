<div align=center>
<img src='./docs/assets/images/logo.png' style="width: 70%">
</div>

|<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/GPU-NVIDIA-%2377b900"> <img src="https://img.shields.io/badge/NPU-Ascend-%23c31d20"> <img src="https://img.shields.io/badge/Python-%3E%3D3.8-%2335709F"> <img src="https://img.shields.io/badge/Docs-Passing-%23009485"> <img src="https://img.shields.io/badge/Codestyle-Black-black"> <img src="https://img.shields.io/badge/PyPI%20Package-0.0.1-%23006DAD"> <img src="https://img.shields.io/badge/🤗Benchmark-HuggingFace-%23FFD21E"> <img src="https://img.shields.io/badge/Pytorch-%3E%3D2.0.0-%23EF5739"> <img src="https://img.shields.io/badge/Hydra-1.3.2-%23E88444"> <img src="https://img.shields.io/badge/Gymnasium-%3E%3D0.28.1-brightgreen"> <img src="https://img.shields.io/badge/DMC Suite-1.0.11-blue"> <img src="https://img.shields.io/badge/Procgen-0.10.7-blueviolet"> <img src="https://img.shields.io/badge/2.2.1-MiniGrid-%23c8c8c8"> <img src="https://img.shields.io/badge/PyBullet-3.2.5-%236A94D4">|
|:-:|

**Hsuanwu: Long-Term Evolution Project of Reinforcement Learning** is inspired by the long-term evolution (LTE) standard project in telecommunications, which aims to track the latest research progress in reinforcement learning (RL) and provide stable and efficient baselines. In Hsuanwu, you can find everything you need in RL, such as training, evaluation, deployment, etc. The highlight features of Hsuanwu:

- ⏱️ Latest algorithms and tricks;
- 🧱 Highly modularized design for complete decoupling of RL algorithms;
- 🚀 Optimized workflow for full hardware acceleration;
- ⚙️ Support for custom environments;
- 🖥️ Support for multiple computing devices like GPU and NPU;
- 🛠️ Support for RL model engineering deployment (TensorRT, CANN, ...);
- 💾 Large number of reusable bechmarks ([See HsuanwuHub](hub.hsuanwu.dev));
- 📋 Elegant experimental management powered by [Hydra](https://hydra.cc/).

See the project structure below:
<div align=center>
<img src='./docs/assets/images/structure.svg' style="width: 100%">
</div>

Please cite the following paper if you use Hsuanwu in your work, thank you!
```bibtex
@article{yuan2023hsuanwu,
  title={Hsuanwu: Long-Term Evolution Project of Reinforcement Learning},
  author={Yuan, Mingqi and Luo, Shihao and Zhang, Zequn and Yang, Xu and Jin, Xin and Li, Bo and Zeng, Wenjun},
  journal={arXiv preprint arXiv:2311.15277},
  year={2023}
}
```

- [Evolution Roadmap](#evolution-roadmap)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Build your first Hsuanwu application](#build-your-first-hsuanwu-application)
- [Implemented Algorithms](#implemented-algorithms)
  - [Learners](#learners)
  - [Intrinsic Reward](#intrinsic-reward)
- [API Documentation](#api-documentation)
- [How To Contribute](#how-to-contribute)
- [Acknowledgment](#acknowledgment)

# Evolution Roadmap
The following figure demonstrates the evolution roadmap of Hsuanwu:
<div align=center>
<img src='./docs/assets/images/roadmap.svg'>
</div>

# Quick Start
## Installation
- with pip `recommended`

Open up a terminal and install Hsuanwu with `pip`:
``` shell
pip install hsuanwu # basic installation
```
or
``` sh
pip install hsuanwu[envs] # for pre-defined environments
```

- with git

Open up a terminal and clone the repository from [GitHub](https://github.com/RLE-Foundation/Hsuanwu) with `git`:
``` sh
git clone https://github.com/RLE-Foundation/Hsuanwu.git
```
After that, run the following command to install package and dependencies:
``` sh
pip install -e . # basic installation
```
or
``` sh
pip install -e .[envs] # for pre-defined environments
```

## Build your first Hsuanwu application
For example, we want to use [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) to solve a task of [DeepMind Control Suite](https://github.com/deepmind/dm_control), and we need the following two steps:

1. Write a `config.yaml` file in your working directory like:
``` yaml
experiment: drqv2_dmc     # Experiment ID.
device: cuda:0            # Device (cpu, cuda, ...) on which the code should be run.
seed: 1                   # Random seed for reproduction.
num_train_steps: 250000   # Number of training steps.

learner:
  name: DrQv2Learner      # The learner name.
```

2. Write a `train.py` file like:
``` python
import hydra # Use Hydra to manage experiments

from hsuanwu.env import make_dmc_env # Import DeepMind Control Suite
from hsuanwu.common.engine import HsuanwuEngine # Import Hsuanwu engine

train_env = make_dmc_env(env_id='cartpole_balance') # Create train env
test_env = make_dmc_env(env_id='cartpole_balance') # Create test env

@hydra.main(version_base=None, config_path='cfgs', config_name='minimum_config')
def main(cfgs):
    engine = HsuanwuEngine(cfgs=cfgs, train_env=train_env, test_env=test_env) # Initialize engine
    engine.invoke() # Start training

if __name__ == '__main__':
    main()
```
Run `train.py` and you will see the following output:

<div align=center>
<img src='./docs/assets/images/rl_training.png'>
</div>

<!-- ``` sh
[04/17/2023 02:03:19 PM] - [HSUANWU INFO ] - Experiment: drqv2_dmc
[04/17/2023 02:03:19 PM] - [HSUANWU INFO ] - Invoking Hsuanwu Engine...
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Checking the Compatibility of Modules...
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Selected Encoder: TassaCnnEncoder
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Selected Learner: DrQv2Learner
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Selected Storage: NStepReplayStorage
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Selected Distribution: TruncatedNormalNoise
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Use Augmentation: True, RandomShift
[04/17/2023 02:03:19 PM] - [HSUANWU DEBUG] - Use Intrinsic Reward: False
[04/17/2023 02:03:19 PM] - [HSUANWU INFO ] - Deploying OffPolicyTrainer...
[04/17/2023 02:03:20 PM] - [HSUANWU DEBUG] - Check Accomplished. Start Training...
[04/17/2023 02:03:30 PM] - [HSUANWU Test ] - S: 0           | E: 0           | L: 500         | R: 340.163     | T: 0:00:11    
[04/17/2023 02:03:35 PM] - [HSUANWU Train] - S: 2000        | E: 3           | L: 500         | R: 370.810     | FPS: 352.900   | T: 0:00:16    
[04/17/2023 02:03:45 PM] - [HSUANWU Train] - S: 2500        | E: 4           | L: 500         | R: 174.664     | FPS: 51.254    | T: 0:00:25    
[04/17/2023 02:03:55 PM] - [HSUANWU Train] - S: 3000        | E: 5           | L: 500         | R: 159.040     | FPS: 51.368    | T: 0:00:35 
.....
``` -->
For more detailed tutorials, see [https://benchmark.hsuanwu.dev/api](https://benchmark.hsuanwu.dev/api).

# Implemented Algorithms
## Learners
|Name|Recurrent|Box|Discrete|Multi Processing|Paper|Citations|
|:-|:-|:-|:-|:-|:-|:-|
|SAC|❌| ✔️ |❌|❌| [Link](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) |5077⭐|
|DrQ|❌| ✔️ |❌|❌| [Link](https://arxiv.org/pdf/2004.13649) |433⭐|
|DDPG|❌| ✔️ |❌|❌| [Link](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------) |11819⭐|
|DrQ-v2|❌| ✔️ |❌|❌| [Link](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com) |100⭐|
|PPO|❌| ✔️ |✔️|✔️| [Link](https://arxiv.org/pdf/1707.06347) |11155⭐|
|DrAC|❌| ✔️ |✔️|✔️| [Link](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf) |29⭐|
|PPG|❌| ✔️ |✔️|✔️| [Link](http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf) |82⭐|
|IMPALA|✔️| ✔️ |✔️|✔️| [Link](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) |1219⭐|

> - `Recurrent`: Support recurrent .
> - `Box`: A N-dimensional box that containes every point in the action space.
> - `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.

## Intrinsic Reward
| Algorithm | Remark | Year  | Paper | Code                                                                                    |
|:-|:-|:-|:-|:-|
| ICM       | Curiosity-driven exploration                 | 2017  | [Curiosity-Driven Exploration by Self-Supervised Prediction](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf)                                                                                                                                                     | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/icm.py)  |
| RND       | Count-based exploration            | 2019  | [Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)                                                                                                                                                                                         | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/rnd.py)  |
| GIRM      | Curiosity-driven exploration            | 2020  | [Intrinsic Reward Driven Imitation Learning via Generative Model](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf)                                                                                                                                                       | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/girm.py) |
| NGU       | Memory-based exploration                | 2020  | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/pdf/2002.06038)                                                                                                                                                                                | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/ngu.py)  |
| RIDE      | Procedurally-generated environment      | 2020  | [RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments](https://arxiv.org/pdf/2002.12292)                                                                                                                                                      | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/ride.py) |
| RE3       | Shannon Entropy Maximization            | 2021  | [State Entropy Maximization with Random Encoders for Efficient Exploration](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf)                                                                                                                                           | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/re3.py)  |
| RISE      | Rényi Entropy Maximization              | 2022  | [Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9802917/)                                                                                                                          | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/rise.py) |
| REVD      | Rényi Divergence Maximization           | 2022  | [Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning](https://openreview.net/pdf?id=V2pw1VYMrDo)                                                                                                                                           | [Link](https://github.com/RLE-Foundation/Hsuanwu/tree/main/hsuanwu/xplore/reward/revd.py) |



For more detiled modules, see [https://benchmark.hsuanwu.dev/tutorials](https://benchmark.hsuanwu.dev/tutorials)

# API Documentation
View our well-designed documentation: [https://docs.hsuanwu.dev/](https://docs.hsuanwu.dev/)

# How To Contribute
Welcome to contribute to this project! Before you begin writing code, please read [CONTRIBUTING.md](https://github.com/RLE-Foundation/Hsuanwu/blob/main/CONTRIBUTING.md) for guide first.

# Acknowledgment
This project is supported by [FUNDING.yml](https://github.com/RLE-Foundation/Hsuanwu/blob/main/.github/FUNDING.yml). Some code of this project is borrowed or inspired by several excellent projects, and we highly appreciate them. See [ACKNOWLEDGMENT.md](https://github.com/RLE-Foundation/Hsuanwu/blob/main/ACKNOWLEDGMENT.md).
