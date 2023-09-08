<div align=center>
<br>
<img src='./assets/images/logo_horizontal.svg' style="width: 75%">
<br>
RLLTE: 强化学习长期演进计划

<h3> <a href=""> 论文 </a> |
<a href="https://docs.rllte.dev/api/"> 文档 </a> |
<a href="https://github.com/hpcaitech/ColossalAI/tree/main/examples"> 示例 </a> |
<a href="https://github.com/RLE-Foundation/rllte/discussions"> 论坛 </a> |
<a href="https://hub.rllte.dev/"> 基线 </a></h3>

<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/GPU-NVIDIA-%2377b900"> <img src="https://img.shields.io/badge/NPU-Ascend-%23c31d20"> <img src="https://img.shields.io/badge/Python-%3E%3D3.8-%2335709F"> <img src="https://img.shields.io/badge/Docs-Passing-%23009485"> <img src="https://img.shields.io/badge/Codestyle-Black-black"> <img src="https://img.shields.io/badge/PyPI-0.0.1-%23006DAD"> <img src="https://img.shields.io/badge/Coverage-98.00%25-green"> 

| [English](README.md) | [中文](docs/README-zh-Hans.md) |

</div>

# Contents
- [概述](#overview)
- [快速入门](#quick-start)
  + [安装](#installation)
  + [快速训练内置算法](#fast-training-with-built-in-algorithms)
    - [运用NVIDIA GPU](#on-nvidia-gpu)
    - [运用HUAWEI NPU](#on-huawei-npu)
  + [三步创建您的强化学习智能体](#three-steps-to-create-your-rl-agent)
  + [算法解耦与模块替代](#algorithm-decoupling-and-module-replacement)
- [功能列表 (部分)](#function-list-part)
  + [强化学习智能体](#rl-agents)
  + [内在奖励模块](#intrinsic-reward-modules)
- [RLLTE 生态环境](#rllte-ecosystem)
- [API 文档](#api-documentation)
- [引用项目](#cite-the-project)
- [如何贡献](#how-to-contribute)
- [致谢](#acknowledgment)

# 概述
受通信领域长期演进（LTE）标准项目的启发，RLLTE旨在提供用于推进RL研究和应用的开发组件和架构。除了提供一流的算法实现外，**RLLTE**还能够充当开发算法的工具包。

<div align="center">
<a href="https://youtu.be/ShVdiHHyXFM" rel="nofollow">
<img src='./assets/images/youtube.png' style="width: 70%">
</a>
<br>
RLLTE简介.
</div>

为什么使用 **RLLTE**?
- 🧬 提供最新的强化学习算法与技巧;
- 🏞️ 提供完整的生态环境，框架支持设计任务，训练模型，评估模型以及部署算法 (TensorRT, CANN, ...);
- 🧱 针对完整解耦算法，提出模块化设计;
- 🚀 提出算法优化，支持硬件加速;
- ⚙️ 支持自定义任务和模块;
- 🖥️ 支持包括GPU和NPU的多种算力设备;
- 💾 大量可重复使用的基准测试 ([rllte-hub](https://hub.rllte.dev));
- 👨‍✈️ 基于大语言模型打造的copilot.

项目结构如下:
<div align=center>
<img src='./assets/images/structure.svg' style="width: 100%">
</div>

有关这些模块的更详细描述，请参阅[API文档](https://docs.rllte.dev/api).

# 快速入门
## 安装
- 前置条件

当前，我们建议使用`Python>=3.8`，用户可以通过以下方式创建虚拟环境：
``` sh
conda create -n rllte python=3.8
```

- 通过 `pip`

打开终端通过`pip`安装 **rllte**:
``` shell
pip install rllte-core # 安装基本模块
pip install rllte-core[envs] # 安装强化学习预设环境
```

- 通过 `git`

开启终端从[GitHub]中复制仓库(https://github.com/RLE-Foundation/rllte):
``` sh
git clone https://github.com/RLE-Foundation/rllte.git
```
在这之后, 运行以下命令行安装所需的包:
``` sh
pip install -e . # 安装基本模块
pip install -e .[envs] # 安装强化学习预设环境
```

更详细的安装说明, 请参阅, [入门指南](https://docs.rllte.dev/getting_started).

## 快速训练内置算法
**RLLTE** 提供了常用强化学习算法的实现，并且附带简单的界面。
### 运用NVIDIA GPU
假如需要通过 [DrQ-v2](https://openreview.net/forum?id=_SJ-_yyes8) 来学习 [DeepMind Control Suite](https://github.com/deepmind/dm_control)任务, 只需编写如下 `train.py`文件:

``` python
# import `env` and `agent` module
from rllte.env import make_dmc_env 
from rllte.agent import DrQv2

if __name__ == "__main__":
    device = "cuda:0"
    # 创建 env, `eval_env` 可选
    env = make_dmc_env(env_id="cartpole_balance", device=device)
    eval_env = make_dmc_env(env_id="cartpole_balance", device=device)
    # 创建 agent
    agent = DrQv2(env=env, eval_env=eval_env, device=device, tag="drqv2_dmc_pixel")
    # 开始训练
    agent.train(num_train_steps=500000, log_interval=1000)
```
运行`train.py`文件，将会得到如下结果:

<div align=center>
<img src='./assets/images/rl_training_gpu.gif' style="filter: drop-shadow(0px 0px 7px #000);">
</div>

### 运用HUAWEI NPU
与上述案例类似, 如果需要在 HUAWEI NPU 上训练智能体, 只需将 `cuda` 替换为 `npu`:
``` python
device = "cuda:0" -> device = "npu:0"
```

## 三步创建您的强化学习智能体
开发者借助**RLLTE**，只需三步就可以实现一个强化学习算法。接下来这个例子将展示如何针对 Atari games 实现 dvantage Actor-Critic (A2C) 算法。 
- 首先, 声明一个原型:
``` py
from rllte.common.prototype import OnPolicyAgent
```
- 其次, 导入必要的模块:
``` py
from rllte.xploit.encoder import MnihCnnEncoder
from rllte.xploit.policy import OnPolicySharedActorCritic
from rllte.xploit.storage import VanillaRolloutStorage
from rllte.xplore.distribution import Categorical
```
- 最后运行该策略的 `.describe` 函数, 运行结果如下:
``` py
OnPolicySharedActorCritic.describe()
# Output:
# ================================================================================
# Name       : OnPolicySharedActorCritic
# Structure  : self.encoder (shared by actor and critic), self.actor, self.critic
# Forward    : obs -> self.encoder -> self.actor -> actions
#            : obs -> self.encoder -> self.critic -> values
#            : actions -> log_probs
# Optimizers : self.optimizers['opt'] -> (self.encoder, self.actor, self.critic)
# ================================================================================
```
这将会展示策略的结构。最后，将上述模块整合到一起并且编辑 `.update` 函数:
``` py
from torch import nn
import torch as th

class A2C(OnPolicyAgent):
    def __init__(self, env, tag, seed, device, num_steps) -> None:
        super().__init__(env=env, tag=tag, seed=seed, device=device, num_steps=num_steps)
        # 创建模块
        encoder = MnihCnnEncoder(observation_space=env.observation_space, feature_dim=512)
        policy = OnPolicySharedActorCritic(observation_space=env.observation_space,
                                           action_space=env.action_space,
                                           feature_dim=512,
                                           opt_class=th.optim.Adam,
                                           opt_kwargs=dict(lr=2.5e-4, eps=1e-5),
                                           init_fn="xavier_uniform"
                                           )
        storage = VanillaRolloutStorage(observation_space=env.observation_space,
                                        action_space=env.action_space,
                                        device=device,
                                        storage_size=self.num_steps,
                                        num_envs=self.num_envs,
                                        batch_size=256
                                        )
        # 设置所有模块
        self.set(encoder=encoder, policy=policy, storage=storage, distribution=Categorical)
    
    def update(self):
        for _ in range(4):
            for batch in self.storage.sample():
                # 评估采样的动作
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(obs=batch.observations, actions=batch.actions)
                # 策略损失
                policy_loss = - (batch.adv_targ * new_log_probs).mean()
                # 价值损失
                value_loss = 0.5 * (new_values.flatten() - batch.returns).pow(2).mean()
                # 更新
                self.policy.optimizers['opt'].zero_grad(set_to_none=True)
                (value_loss * 0.5 + policy_loss - entropy * 0.01).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy.optimizers['opt'].step()
```
然后, 通过以下方法是训练该智能体
``` py
from rllte.env import make_atari_env
if __name__ == "__main__":
    device = "cuda"
    env = make_atari_env("PongNoFrameskip-v4", num_envs=8, seed=0, device=device)
    agent = A2C(env=env, tag="a2c_atari", seed=0, device=device, num_steps=128)
    agent.train(num_train_steps=10000000)
```
通过如上例子可以发现,  利用 **RLLTE** 只需少数几行代码便可以得到一个强化学习智能体. 

## 算法解耦与模块替代
**RLLTE** 许可开发者将预设好的模块替换, 以便于进行算法性能比较和优化。开发者可以将预设模块替换成别的类型的内置模块或者自定义模块。假设我们想要对比不同编码器的效果. 祝需要调用其中 `.set` 函数:
``` py
from rllte.xploit.encoder import EspeholtResidualEncoder
encoder = EspeholtResidualEncoder(...)
agent.set(encoder=encoder)
```
**RLLTE** 框架简便, 给予开发者们最大程度的自由。更多详细说明, 请参考 [教程](https://docs.rllte.dev/tutorials)。

# 功能列表 (部分)
## 强化学习智能体
|     类型    |  算法 | 连续状态 | 离散状态 | 多重二进制 | 多重离散 | 多线程 | NPU |💰|🔭|
|:-----------:|:------:|:---:|:----:|:----:|:----:|:------:|:---:|:------:|:---:|
| On-Policy   | [A2C](https://arxiv.org/abs/1602.01783)    | ✔️   | ✔️    | ✔️    | ✔️    | ✔️    | ✔️   |✔️    |❌    |
| On-Policy   | [PPO](https://arxiv.org/pdf/1707.06347)    | ✔️   | ✔️    | ✔️    | ✔️    | ✔️    | ✔️   |✔️    |❌    |
| On-Policy   | [DrAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)| ✔️   | ✔️    | ✔️    | ✔️    | ✔️    | ✔️   |✔️    | ✔️   |
| On-Policy   | [DAAC](http://proceedings.mlr.press/v139/raileanu21a/raileanu21a.pdf)| ✔️   | ✔️    | ✔️    | ✔️    | ✔️    | ✔️   |✔️    | ❌   |
| On-Policy   | [DrDAAC](https://proceedings.neurips.cc/paper/2021/file/2b38c2df6a49b97f706ec9148ce48d86-Paper.pdf)| ✔️   | ✔️    | ✔️    | ✔️    | ✔️    | ✔️   |✔️    | ✔️   |
| On-Policy   | [PPG](http://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf)| ✔️   | ✔️    | ✔️    |  ❌   | ✔️    | ✔️   |✔️    | ❌   |
| Off-Policy  | [DQN](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf) | ✔️   | ❌    | ❌    | ❌    | ✔️    | ✔️   |✔️    | ❌   |
| Off-Policy  | [DDPG](https://arxiv.org/pdf/1509.02971.pdf?source=post_page---------------------------)| ✔️   | ❌    | ❌    | ❌    | ✔️    | ✔️   |✔️    |❌    |
| Off-Policy  | [SAC](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)| ✔️   | ❌    | ❌    | ❌    | ✔️    | ✔️   |✔️    |❌    |
| Off-Policy  | [TD3](http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf)| ✔️   | ❌    | ❌    | ❌    | ✔️    | ✔️   |✔️    |❌    |
| Off-Policy  | [DrQ-v2](https://arxiv.org/pdf/2107.09645.pdf?utm_source=morioh.com)| ✔️   | ❌    | ❌    | ❌    | ❌    | ✔️   |✔️    |✔️    |
| Distributed | [IMPALA](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) | ✔️   | ✔️    | ❌    | ❌    | ✔️    | ❌   |❌    |❌    |

> - 🐌: 开发中;
> - 💰: 支持 intrinsic reward shaping;
> - 🔭: 支持 observation augmentation。


## 内在奖励模块
| **类型** 	| **模块** 	|
|---	|---	|
| Count-based 	| [PseudoCounts](https://arxiv.org/pdf/2002.06038), [RND](https://arxiv.org/pdf/1810.12894.pdf) 	|
| Curiosity-driven 	| [ICM](http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf), [GIRM](http://proceedings.mlr.press/v119/yu20d/yu20d.pdf), [RIDE](https://arxiv.org/pdf/2002.12292) 	|
| Memory-based 	| [NGU](https://arxiv.org/pdf/2002.06038) 	|
| Information theory-based 	| [RE3](http://proceedings.mlr.press/v139/seo21a/seo21a.pdf), [RISE](https://ieeexplore.ieee.org/abstract/document/9802917/), [REVD](https://openreview.net/pdf?id=V2pw1VYMrDo) 	|

详细案例请参考 [Tutorials: Use Intrinsic Reward and Observation Augmentation](https://docs.rllte.dev/tutorials/data_augmentation).

# RLLTE 生态环境
欢迎使用RLLTE 生态环境, 为您打造舒适便利的平台:

- [Hub](https://docs.rllte.dev/benchmarks/): 提供快速训练的 API 接口以及可重复使用的基准测试.
- [Evaluation](https://docs.rllte.dev/api/tutorials/): 提供可信赖的模型评估标准。
- [Env](https://docs.rllte.dev/api/tutorials/): 提供封装完善的环境。
- [Deployment](https://docs.rllte.dev/api/tutorials/): 提供便捷的算法部署接口。
- [Pre-training](https://docs.rllte.dev/api/tutorials/): 提供多种强化学习预训练的方式。
- [Copilot](https://docs.rllte.dev/copilot): 提供大语言模型 copilot。

# API 文档
请参阅我们详细的 API 文档: [https://docs.rllte.dev/](https://docs.rllte.dev/)
<div align=center>
<img src='./assets/images/docs.gif' style="width: 100%">
</div>

# 如何贡献
欢迎参与贡献我们的项目! 在您准备之前, 请先参阅[CONTRIBUTING.md](https://github.com/RLE-Foundation/rllte/blob/main/CONTRIBUTING.md)。

# 引用项目
如果您想在研究中引用 **RLLTE**, 请参考如下格式:
``` tex
@software{rllte,
  author = {Mingqi Yuan, Zequn Zhang, Yang Xu, Shihao Luo, Bo Li, Xin Jin, and Wenjun Zeng},
  title = {RLLTE: Long-Term Evolution Project of Reinforcement Learning},
  url = {https://github.com/RLE-Foundation/rllte},
  year = {2023},
}
```

# 致谢
该项目由 [香港理工大学](http://www.polyu.edu.hk/), [东方理工高等研究院](http://www.eias.ac.cn/), 以及 [FLW-Foundation](FLW-Foundation)赞助。 [东方理工高等研究院](https://hpc.eias.ac.cn/) 提供 GPU 计算平台, [华为异腾](https://www.hiascend.com/) 提供 NPU 计算平台。 该项目的部分代码参考了其他优秀的开源项目. 请参见 [ACKNOWLEDGMENT.md](https://github.com/RLE-Foundation/rllte/blob/main/ACKNOWLEDGMENT.md).
