<div align=center>
<img src='./docs/assets/images/rllte-logo.png' style="width: 40%">
</div>

|<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/GPU-NVIDIA-%2377b900"> <img src="https://img.shields.io/badge/NPU-Ascend-%23c31d20"> <img src="https://img.shields.io/badge/Python-%3E%3D3.8-%2335709F"> <img src="https://img.shields.io/badge/Docs-Passing-%23009485"> <img src="https://img.shields.io/badge/Codestyle-Black-black"> <img src="https://img.shields.io/badge/PyPI%20Package-0.0.1-%23006DAD"> <img src="https://img.shields.io/badge/🤗Benchmark-HuggingFace-%23FFD21E"> <img src="https://img.shields.io/badge/Pytorch-%3E%3D2.0.0-%23EF5739"> <img src="https://img.shields.io/badge/Hydra-1.3.2-%23E88444"> <img src="https://img.shields.io/badge/Gymnasium-%3E%3D0.28.1-brightgreen"> <img src="https://img.shields.io/badge/DMC Suite-1.0.11-blue"> <img src="https://img.shields.io/badge/Procgen-0.10.7-blueviolet"> <img src="https://img.shields.io/badge/2.2.1-MiniGrid-%23c8c8c8"> <img src="https://img.shields.io/badge/PyBullet-3.2.5-%236A94D4"> <img src="https://img.shields.io/badge/Robosuite-1.4.0-%23b51800">|
|:-:|

**RLLTE: Long-Term Evolution Project of Reinforcement Learning** is inspired by the long-term evolution (LTE) standard project in telecommunications, which aims to track the latest research progress in reinforcement learning (RL) and provide stable and efficient baselines. In **rllte**, you can find everything you need in RL, such as training, evaluation, deployment, etc. 

If you use **rllte** in your research, please cite this project like this:
``` tex
@software{rllte,
  author = {Mingqi Yuan, Zequn Zhang, Yang Xu, Shihao Luo, Bo Li, Xin Jin, and Wenjun Zeng},
  title = {RLLTE: Long-Term Evolution Project of Reinforcement Learning},
  url = {https://github.com/RLE-Foundation/rllte},
  year = {2023},
}
```

# Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Implemented Modules](#implemented-modules)
- [Benchmark](#benchmark)
- [API Documentation](#api-documentation)
- [How To Contribute](#how-to-contribute)
- [Acknowledgment](#acknowledgment)

# Overview
For the project tenet, please read [Evolution Tenet](https://docs.rllte.dev/tenet).

The highlight features of **rllte**:
- 👨‍✈️ Large language model-empowered copilot;
- ⏱️ Latest algorithms and tricks;
- 📕 Standard and sophisticated modules for redevelopment;
- 🧱 Highly modularized design for complete decoupling of RL algorithms;
- 🚀 Optimized workflow for full hardware acceleration;
- ⚙️ Support custom environments and modules;
- 🖥️ Support multiple computing devices like GPU and NPU;
- 🛠️ Support RL model engineering deployment (TensorRT, CANN, ...);
- 💾 Large number of reusable benchmarks (See [rllte-hub](https://hub.rllte.dev));

See the project structure below:
<div align=center>
<img src='./docs/assets/images/structure.svg' style="width: 100%">
</div>

- **[Common](https://docs.rllte.dev/common_index/)**: Base classes and auxiliary modules like logger and timer.

- **[Xploit](https://docs.rllte.dev/xploit_index/)**: Modules that focus on <font color="#B80000"><b>exploitation</b></font> in RL.
    + **Encoder**: *Neural nework-based encoders for processing observations.*
    + **Agent**: *Agents for interacting and learning.*
    + **Storage**: *Storages for storing collected experiences.*

- **[Xplore](https://docs.rllte.dev/xplore_index/)**: Modules that focus on <font color="#B80000"><b>exploration</b></font> in RL.
    + **Augmentation**: *PyTorch.nn-like modules for observation augmentation.*
    + **Distribution**: *Distributions for sampling actions.*
    + **Reward**: *Intrinsic reward modules for enhancing exploration.*

- **[Hub]()**: Reusable datasets and models.

- **[Env](https://docs.rllte.dev/env_index/)**: Packaged environments (e.g., Atari games) for fast invocation.

- **[Evaluation](https://docs.rllte.dev/evaluation_index/)**: Reasonable and reliable metrics for algorithm evaluation.

- **[Pre-training](https://docs.rllte.dev/pretraining_index/)**: Methods of <font color="#B80000"><b>pre-training</b></font> in RL.

- **[Deployment](https://docs.rllte.dev/deployment_index/)**: Methods of <font color="#B80000"><b>model deployment</b></font> in RL.

For more detiled descriptions of these modules, see [https://docs.rllte.dev/api](https://docs.rllte.dev/api)

# Quick Start
## Installation
- Prerequisites

Currently, we recommend `Python>=3.8`, and user can create an virtual environment by
``` sh
conda create -n rllte python=3.8
```

- with pip `recommended`

Open up a terminal and install **rllte** with `pip`:
``` shell
pip install rllte # basic installation
pip install rllte[envs] # for pre-defined environments
```

- with git

Open up a terminal and clone the repository from [GitHub](https://github.com/RLE-Foundation/rllte) with `git`:
``` sh
git clone https://github.com/RLE-Foundation/rllte.git
```
After that, run the following command to install package and dependencies:
``` sh
pip install -e . # basic installation
pip install -e .[envs] # for pre-defined environments
```

For more detailed installation instruction, see [https://docs.rllte.dev/getting_started](https://docs.rllte.dev/getting_started).

## Start Training
### On NVIDIA GPU
For example, we want to use [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) to solve a task of [DeepMind Control Suite](https://github.com/deepmind/dm_control), and it suffices to write a `train.py` like:

``` python
# import `env` and `agent` api
from rllte.env import make_dmc_env 
from rllte.xploit.agent import DrQv2

if __name__ == "__main__":
    device = "cuda:0"
    # create env, `eval_env` is optional
    env = make_dmc_env(env_id="cartpole_balance", device=device)
    eval_env = make_dmc_env(env_id="cartpole_balance", device=device)
    # create agent
    agent = DrQv2(env=env, 
                  eval_env=eval_env, 
                  device='cuda',
                  tag="drqv2_dmc_pixel")
    # start training
    agent.train(num_train_steps=5000)
```
Run `train.py` and you will see the following output:

<div align=center>
<img src='./docs/assets/images/rl_training_gpu.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

### On HUAWEI NPU
Similarly, if we want to train an agent on HUAWEI NPU, it suffices to replace `DrQv2` with `NpuDrQv2`:
``` python
# import `env` and `agent` api
from rllte.env import make_dmc_env 
from rllte.xploit.agent import DrQv2

if __name__ == "__main__":
    device = "npu:0"
    # create env, `eval_env` is optional
    env = make_dmc_env(env_id="cartpole_balance", device=device)
    eval_env = make_dmc_env(env_id="cartpole_balance", device=device)
    # create agent
    agent = DrQv2(env=env, 
                  eval_env=eval_env, 
                  device='cuda',
                  tag="drqv2_dmc_pixel")
    # start training
    agent.train(num_train_steps=5000)
```
Then you will see the following output:
<div align=center>
<img src='./docs/assets/images/rl_training_npu.png' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

> Please refer to [Implemented Modules](#implemented-modules) for the compatibility of NPU.

For more detailed tutorials, see [https://docs.rllte.dev/tutorials](https://docs.rllte.dev/tutorials).

# Implemented Modules
## RL Agents
<!-- |Module|Recurrent|Box|Discrete|MultiBinary|Multi Processing|NPU|Paper|Citations|
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
|SAC|❌| ✔️ |❌|❌|❌|✔️ | [Link](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf) |5077⭐|
|DrQ|❌| ✔️ |❌|❌|❌|✔️ | [Link](https://arxiv.org/pdf/2004.13649) |433⭐|
|DDPG|❌| ✔️ |❌|❌|❌|✔️ | [Link](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------) |11819⭐|
|DrQ-v2|❌| ✔️ |❌|❌|❌|✔️ | [Link](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com) |100⭐|
|PPO|❌| ✔️ |✔️|✔️|✔️|✔️ | [Link](https://arxiv.org/pdf/1707.06347) |11155⭐|
|DrAC|❌| ✔️ |✔️|✔️|✔️|✔️ | [Link](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf) |29⭐|
|DAAC|❌| ✔️ |✔️|✔️|✔️|✔️ | [Link](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf) |56⭐|
|IMPALA|✔️| ✔️ |✔️|❌|✔️|🐌| [Link](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) |1219⭐| -->

<table>
<thead>
  <tr>
    <th>Type</th>
    <th>Module</th>
    <th>Recurrent</th>
    <th>Box</th>
    <th>Discrete</th>
    <th>MultiBinary</th>
    <th>Multi Processing</th>
    <th>NPU</th>
    <th>Paper</th>
    <th>Citations</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">Original</td>
    <td>SAC</td>
    <td>❌</td>
    <td>✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>✔️</td>
    <td><a href="http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>5077⭐</td>
  </tr>
  <tr>
    <td>DDPG</td>
    <td>❌</td>
    <td>✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>✔️</td>
    <td><a href="https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>11819⭐</td>
  </tr>
  <tr>
    <td>PPO</td>
    <td>❌</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td><a href="https://arxiv.org/pdf/1707.06347" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>11155⭐</td>
  </tr>
  <tr>
    <td>DAAC</td>
    <td>❌</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td><a href="http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>56⭐</td>
  </tr>
  <tr>
    <td>IMPALA</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>❌</td>
    <td>✔️</td>
    <td>❌</td>
    <td><a href="http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>1219⭐</td>
  </tr>
  <tr>
    <td rowspan="3">Augmented</td>
    <td>DrQ-v2</td>
    <td>❌</td>
    <td>✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>✔️</td>
    <td><a href="https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>100⭐</td>
  </tr>
  <tr>
    <td>DrQ</td>
    <td>❌</td>
    <td>✔️</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
    <td>✔️</td>
    <td><a href="https://arxiv.org/pdf/2004.13649" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>433⭐</td>
  </tr>
  <tr>
    <td>DrAC</td>
    <td>❌</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td>✔️</td>
    <td><a href="https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf" target="_blank" rel="noopener noreferrer">Link</a></td>
    <td>29⭐</td>
  </tr>
</tbody>
</table>

> - DrQ=SAC+Augmentation, DDPG=DrQ-v2-Augmentation, DrAC=PPO+Augmentation.
> - 🐌: Developing.
> - `NPU`: Support Neural-network processing unit.
> - `Recurrent`: Support recurrent neural network.
> - `Box`: A N-dimensional box that containes every point in the action space.
> - `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.
> - `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.

## Intrinsic Reward Modules
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

> - 🐌: Developing.
> - `Repr.`: The method involves representation learning.
> - `Visual`: The method works well in visual RL.

See [Tutorials: Use Intrinsic Reward and Observation Augmentation](https://docs.rllte.dev/tutorials/data_augmentation) for usage examples.

# Benchmark
**rllte** provides a large number of reusable bechmarks, see [https://hub.rllte.dev/](https://hub.rllte.dev/) and [https://docs.rllte.dev/benchmarks/](https://docs.rllte.dev/benchmarks/)

# API Documentation
View our well-designed documentation: [https://docs.rllte.dev/](https://docs.rllte.dev/)

# How To Contribute
Welcome to contribute to this project! Before you begin writing code, please read [CONTRIBUTING.md](https://github.com/RLE-Foundation/rllte/blob/main/CONTRIBUTING.md) for guide first.

# Acknowledgment
This project is supported by [The Hong Kong Polytechnic University](http://www.polyu.edu.hk/), [Eastern Institute for Advanced Study](http://www.eias.ac.cn/), and [FLW-Foundation](FLW-Foundation). [EIAS HPC](https://hpc.eias.ac.cn/) provides a GPU computing platform, and [Ascend Community](https://www.hiascend.com/) provides an NPU computing platform for our testing. Some code of this project is borrowed or inspired by several excellent projects, and we highly appreciate them. See [ACKNOWLEDGMENT.md](https://github.com/RLE-Foundation/rllte/blob/main/ACKNOWLEDGMENT.md).
