<div align=center>
<img src='./docs/assets/images/logo.png' style="width: 70%">
</div>

<img src="https://img.shields.io/badge/License-Apache-%23CA2136"> <img src="https://img.shields.io/badge/Python-%3E%3D3.8-%2335709F"> <img src="https://img.shields.io/badge/Docs-Passing-%23009485"> <img src="https://img.shields.io/badge/Codestyle-Black-black"> <img src="https://img.shields.io/badge/PyPI%20Package-0.0.1-%23006DAD"> <img src="https://img.shields.io/badge/Pytorch-%3E%3D1.12.0-%23EF5739"> <img src="https://img.shields.io/badge/Hydra-1.3.2-%23E88444"> <img src="https://img.shields.io/badge/Gym-%3E%3D0.21.0-brightgreen"> <img src="https://img.shields.io/badge/DMC Suite-1.0.5-blue"> <img src="https://img.shields.io/badge/Procgen-0.10.7-blueviolet"> <img src="https://img.shields.io/badge/PyBullet-3.2.5-%236A94D4">

**Hsuanwu: Long-Term Evolution Project of Reinforcement Learning** is inspired by the long-term evolution (LTE) standard project in telecommunications, which aims to track the latest research progress in reinforcement learning (RL) and provide stable and efficient baselines. The highlight features of Hsuanwu:


- 🧱 Complete decoupling of RL algorithms, and each module can be invoked separately;
- 📚 Large number of reusable bechmarking results ([See Benchmarks](benchmark.hsuanwu.dev));
- 🛠️ Support for RL model engineering deployment (C++ API);
- 🚀 Minimizing the CPU to GPU data transferring to realize full GPU-acceleration;
- 📋 Elegant experimental management powered by [Hydra](https://hydra.cc/).

See the project structure below:
<div align=center>
<img src='./docs/assets/images/structure.png' style="width: 70%">
</div>

# Qucik Start
## Installation
Open up a terminal and install Hsuanwu with `pip`:
```
pip install hsuanwu
```
## Build your first Hsuanwu application
Copy the `config.yaml` file to your working directory:

Write a `train.py` file like:
``` python
from hsuanwu.env import make_dmc_env
from hsuanwu.common.engine import OffPolicyTrainer

train_env = make_dmc_env(env_id='cartpole_balance')
test_env = make_dmc_env(env_id='cartpole_balance')

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(cfgs):
    trainer = OffPolicyTrainer(train_env=train_env, test_env=test_env, cfgs=cfgs)
    trainer.train()

if __name__ == '__main__':
    main()
```
Run `train.py` and you will see the following output:


# API Documentation
View our well-designed documentation: [https://docs.hsuanwu.dev/](https://docs.hsuanwu.dev/)

# How To Contribute
Welcome to contribute to this project! Before you begin writing code, please read [CONTRIBUTING.md](https://github.com/RLE-Foundation/Hsuanwu/blob/main/CONTRIBUTING.md) for guide first.

# Acknowledgment
