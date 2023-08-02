# Getting Started

## Installation

### Prerequisites
Currently, we recommend `Python>=3.8`, and user can create an virtual environment by
``` sh
conda create -n rllte python=3.8
```

### with pip <small>recommended</small>
**RLLTE** has been published as a Python package in [PyPi](https://pypi.org/project/rllte/) and can be installed with `pip`, ideally by using a virtual environment. Open up a terminal and install **RLLTE** with:

``` shell
pip install rllte-core # basic installation
pip install rllte-core[envs] # for pre-defined environments
```

### with git
Open up a terminal and clone the repository from [GitHub](https://github.com/RLE-Foundation/rllte) witg `git`:
``` sh
git clone https://github.com/RLE-Foundation/rllte.git
```
After that, run the following command to install package and dependencies:
``` sh
pip install -e . # basic installation
pip install -e .[envs] # for pre-defined environments
```

## PyTorch Installation
**RLLTE** currently supports two kinds of computing devices for acceleration, namely [NVIDIA GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/) and [HUAWEI NPU](https://www.hiascend.com/). Thus users need to install different versions PyTorch for adapting to different devices.
### with NVIDIA GPU
Open up a terminal and install PyTorch with:
``` sh
pip3 install torch==2.0.0 torchvision
```
More information can be found in [Get Started](https://pytorch.org/get-started/locally/).

### with HUAWEI NPU
!!! tip
    Ascend NPU only supports aarch64!

- Install the dependencies for PyTorch:
``` sh
pip3 install pyyaml wheel
```
- Download the `.whl` package of PyTorch from Kunpeng file sharing center and install it:
``` sh
wget https://repo.huaweicloud.com/kunpeng/archive/Ascend/PyTorch/torch-1.11.0-cp39-cp39-linux_aarch64.whl
pip3 install torch-1.11.0-cp39-cp39-linux_aarch64.whl
```

- Install `torch_npu`:
``` sh
wget https://gitee.com/ascend/pytorch/releases/download/v5.0.rc1-pytorch1.11.0/torch_npu-1.11.0-cp39-cp39m-linux_aarch64.whl
pip3 install torch_npu-1.11.0-cp39-cp39m-linux_aarch64.whl
```

- Install `apex` [Optional]:
```
wget https://gitee.com/ascend/apex/releases/download/v5.0.rc1-pytorch1.11.0/apex-0.1_ascend-cp39-cp39m-linux_aarch64.whl
pip3 install apex-0.1_ascend-cp39-cp39m-linux_aarch64.whl
```
Training with mixed precision can improve the model performance. You can introduce the Apex mixed precision module or use the AMP module integrated in AscendPyTorch 1.8.1 or later based on the scenario. The Apex module provides four function modes to suit different training with mixed precision scenarios. AMP is only similar to one function of the Apex module, but can be directly used without being introduced. For details about how to use the AMP and Apex modules, see "Mixed Precision Description" in the [PyTorch Network Model Porting and Training Guide](https://www.hiascend.com/document/detail/en/canncommercial/601/modeldevpt/ptmigr/ptmigr_0001.html).