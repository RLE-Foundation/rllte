<div align=center>
<br>
<img src='./docs/assets/images/logo_horizontal.svg' style="width: 75%">
<br>
RLLTE: Long-Term Evolution Project of Reinforcement Learning

<h3> <a href=""> Paper </a> |
<a href="https://docs.rllte.dev/api/"> Documentation </a> |
<a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples"> Tutorials </a> |
<a href="https://github.com/RLE-Foundation/rllte/discussions"> Forum </a> |
<a href="https://hub.rllte.dev/"> Benchmarks </a></h3>

<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/GPU-NVIDIA-%2377b900"> <img src="https://img.shields.io/badge/NPU-Ascend-%23c31d20"> <img src="https://img.shields.io/badge/Python-%3E%3D3.8-%2335709F"> <img src="https://img.shields.io/badge/Docs-Passing-%23009485"> <img src="https://img.shields.io/badge/Codestyle-Black-black"> <img src="https://img.shields.io/badge/PyPI-0.0.1-%23006DAD"> <img src="https://img.shields.io/badge/Coverage-98.00%25-green"> 

| [English](README.md) | [中文](docs/README-zh-Hans.md) |

</div>

# Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Implemented Modules (Part)](#implemented-modules-part)
- [Benchmarks](#benchmarks)
- [API Documentation](#api-documentation)
- [Cite the Project](#cite-the-project)
- [How To Contribute](#how-to-contribute)
- [Acknowledgment](#acknowledgment)

# Overview
Inspired by the long-term evolution (LTE) standard project in telecommunications, aiming to provide development components for and standards for advancing RL research and applications. **RLLTE** is **not** designed to provide specific RL algorithms but a toolkit for producing algorithms.

<div align="center">
<a href="https://youtu.be/ShVdiHHyXFM" rel="nofollow">
<img src='./docs/assets/images/youtube.png' style="width: 70%">
</a>
<br>
An introduction to RLLTE.
</div>

Why **RLLTE**?
- 🧬 Long-term evolution for providing latest algorithms and tricks;
- 🏞️ Complete ecosystem for task design, model training, evaluation, and deployment (TensorRT, CANN, ...);
- 🧱 Module-oriented design for complete decoupling of RL algorithms;
- 🚀 Optimized workflow for full hardware acceleration;
- ⚙️ Support custom environments and modules;
- 🖥️ Support multiple computing devices like GPU and NPU;
- 💾 Large number of reusable benchmarks (See [rllte-hub](https://hub.rllte.dev));
- 👨‍✈️ Large language model-empowered copilot.

See the project structure below:
<div align=center>
<img src='./docs/assets/images/structure.svg' style="width: 100%">
</div>

For more detiled descriptions of these modules, see [API Documentation](https://docs.rllte.dev/api).

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
pip install rllte-core # basic installation
pip install rllte-core[envs] # for pre-defined environments
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

For more detailed installation instruction, see [Getting Started](https://docs.rllte.dev/getting_started).

## Start Training
### On NVIDIA GPU
For example, we want to use [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) to solve a task of [DeepMind Control Suite](https://github.com/deepmind/dm_control), and it suffices to write a `train.py` like:

``` python
# import `env` and `agent` api
from rllte.env import make_dmc_env 
from rllte.agent import DrQv2

if __name__ == "__main__":
    device = "cuda:0"
    # create env, `eval_env` is optional
    env = make_dmc_env(env_id="cartpole_balance", device=device)
    # create agent
    agent = DrQv2(env=env, device=device, tag="drqv2_dmc_pixel")
    # start training
    agent.train(num_train_steps=500000)
```
Run `train.py` and you will see the following output:

<div align=center>
<img src='./docs/assets/images/rl_training_gpu.gif' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

### On HUAWEI NPU
Similarly, if we want to train an agent on HUAWEI NPU, it suffices to replace `cuda` with `npu`:
``` python
device = "cuda:0" -> device = "npu:0"
```

Please refer to [Implemented Modules](#implemented-modules-part) for the compatibility of NPU. For more detailed tutorials, see [Tutorials](https://docs.rllte.dev/tutorials).

# Implemented Modules (Part)
## RL Agents
| Type 	| Legacy 	| Current 	|
|---	|---	|---	|
| On-Policy 	| [A2C](https://arxiv.org/abs/1602.01783)<sup>🖥️⛓️💰</sup>,[PPO](https://arxiv.org/pdf/1707.06347)<sup>🖥️⛓️💰</sup> 	| [DAAC](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf)<sup>🖥️⛓️💰</sup>,[DrAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)<sup>🖥️⛓️💰🔭</sup>,[DrDAAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)<sup>🖥️⛓️💰🔭</sup> 	|
| Off-Policy 	| [DQN](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)<sup>🖥️⛓️💰</sup>,[DDPG](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------)<sup>🖥️⛓️💰</sup>,[SAC](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)<sup>🖥️⛓️💰</sup> 	| [DrQ-v2](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)<sup>🖥️⛓️💰🔭</sup> 	|
| Distributed 	|  	| [IMPALA](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)<sup>⛓️</sup> 	|

> - 🖥️: Support Neural-network processing unit.
> - ⛓️: Multi Processing.
> - 💰: Support intrinsic reward shaping.
> - 🔭: Support observation augmentation.


## Intrinsic Reward Modules
| **Type** 	| **Modules** 	|
|---	|---	|
| Count-based 	| [PseudoCounts](https://arxiv.org/pdf/2002.06038), [RND](https://arxiv.org/pdf/1810.12894.pdf) 	|
| Curiosity-driven 	| [ICM](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf), [GIRM](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf), [RIDE](https://arxiv.org/pdf/2002.12292) 	|
| Memory-based 	| [NGU](https://arxiv.org/pdf/2002.06038) 	|
| Information theory-based 	| [RE3](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf), [RISE](https://ieeexplore.ieee.org/abstract/document/9802917/), [REVD](https://openreview.net/pdf?id=V2pw1VYMrDo) 	|

See [Tutorials: Use Intrinsic Reward and Observation Augmentation](https://docs.rllte.dev/tutorials/data_augmentation) for usage examples.

# Benchmarks
**RLLTE** provides a large number of reusable bechmarks, see [https://hub.rllte.dev/](https://hub.rllte.dev/) and [https://docs.rllte.dev/benchmarks/](https://docs.rllte.dev/benchmarks/)

# API Documentation
View our well-designed documentation: [https://docs.rllte.dev/](https://docs.rllte.dev/)
<div align=center>
<img src='./docs/assets/images/docs.gif' style="width: 100%">
</div>

# How To Contribute
Welcome to contribute to this project! Before you begin writing code, please read [CONTRIBUTING.md](https://github.com/RLE-Foundation/rllte/blob/main/CONTRIBUTING.md) for guide first.

# Cite the Project
If you use **RLLTE** in your research, please cite this project like this:
``` tex
@software{rllte,
  author = {Mingqi Yuan, Zequn Zhang, Yang Xu, Shihao Luo, Bo Li, Xin Jin, and Wenjun Zeng},
  title = {RLLTE: Long-Term Evolution Project of Reinforcement Learning},
  url = {https://github.com/RLE-Foundation/rllte},
  year = {2023},
}
```

# Acknowledgment
This project is supported by [The Hong Kong Polytechnic University](http://www.polyu.edu.hk/), [Eastern Institute for Advanced Study](http://www.eias.ac.cn/), and [FLW-Foundation](FLW-Foundation). [EIAS HPC](https://hpc.eias.ac.cn/) provides a GPU computing platform, and [HUAWEI Ascend Community](https://www.hiascend.com/) provides an NPU computing platform for our testing. Some code of this project is borrowed or inspired by several excellent projects, and we highly appreciate them. See [ACKNOWLEDGMENT.md](https://github.com/RLE-Foundation/rllte/blob/main/ACKNOWLEDGMENT.md).
