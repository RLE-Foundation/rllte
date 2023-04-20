# Getting Started

## Installation

### Prerequisites
Currently, Hsuanwu requires `Python>=3.8`, user can create an virtual environment by
``` sh
conda create -n hsuanwu python=3.8
```

### with pip <small>recommended</small>
**Hsuanwu** has been published as a Python package in [PyPi](https://pypi.org/project/hsuanwu/) and can be installed with `pip`, ideally by using a virtual environment. Open up a terminal and install **Hsuanwu** with:

``` sh
pip install hsuanwu # basic installation
pip install hsuanwu[envs] # for pre-defined environments
pip install hsuanwu[tests] # for project tests
pip install hsuanwu[all] # install all the dependencies
```

### with git
Open up a terminal and clone the repository from [GitHub](https://github.com/RLE-Foundation/Hsuanwu) witg `git`:
``` sh
git clone https://github.com/RLE-Foundation/Hsuanwu.git
```
After that, run the following command to install package and dependencies:
``` sh
pip install -e . # basic installation
pip install -e .[envs] # for pre-defined environments
pip install -e .[tests] # for project tests
pip install -e .[all] # install all the dependencies
```

## PyTorch Installation
**Hsuanwu** currently supports two kinds of computing devices for acceleration, namely [NVIDIA GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/) and [HUAWEI NPU](https://www.hiascend.com/). Thus users need to install different versions PyTorch for adapting to different devices.
### with NVIDIA GPU
Open up a terminal and install PyTorch with:
``` sh
pip3 install torch torchvision
```
More information can be found in [Get Started](https://pytorch.org/get-started/locally/).

!!! info
    Hsuanwu now supports PyTorch 2.0.0!

### with HUAWEI NPU
