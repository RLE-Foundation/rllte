<div align=center>
<br>
<img src='./assets/images/hub_logo.png' style="width: 85%">
<br>

RLLTE Hub: Large-Scale and Comprehensive Data Hub for RL
</div>

# Contents
- [Overview](#overview)
- [Installation](#installation)
- [We Provide](#we-provide)
  - [Trained RL Models](#trained-rl-models)
  - [RL Training Datasets](#rl-training-datasets)
  - [RL Training Applications](#rl-training-applications)
  - [Offline RL Datasets](#offline-rl-datasets)

# Overview
**RLLTE Hub** is a repository of multifarious trained models and datasets of reinforcement learning (RL).

| **Module** | **Remark**|
|:--|:--|
|📊 `rllte.hub.datasets`|	Provide test scores and learning cures of various RL algorithms on different benchmarks.|
|🗃️ `rllte.hub.models`|	Provide trained models of various RL algorithms on different benchmarks.|
|🎮 `rllte.hub.applications`|	Provide fast-APIs for training RL agents on recognized benchmarks.|

# Installation
Open up a terminal and install `rllte-hub` with `pip`:
```
pip install rllte-core
```

# We Provide
## Trained RL Models
## RL Training Logs
Download training logs of various RL algorithms on well-recognized benchmarks for academic research.
## RL Training Applications
## Offline RL Datasets
The dataset structure:
``` txt
demonstrations
├── expert
│   ├── Breakout-v5
│   │   ├── episode_0
│   │   │   ├── observations
│   │   │   ├── actions
│   │   │   ├── rewards
│   │   │   ├── terminateds
│   │   │   └── truncateds
│   │   ├── episode_1
│   │   │   ├── observations
│   │   │   ├── actions
│   │   │   ├── rewards
│   │   │   ├── terminateds
│   │   │   └── truncateds
│   │   └── ...
│   ├── Pong-v5
│   │   └── ...
│   └── ...
└── random
    └── ...
```

# Cite the Project
If you use this project in your research, please cite this project like this:

``` bibtex
@article{yuan2023rllte,
  title={RLLTE: Long-Term Evolution Project of Reinforcement Learning}, 
  author={Mingqi Yuan and Zequn Zhang and Yang Xu and Shihao Luo and Bo Li and Xin Jin and Wenjun Zeng},
  year={2023},
  journal={arXiv preprint arXiv:2309.16382}
}
```
