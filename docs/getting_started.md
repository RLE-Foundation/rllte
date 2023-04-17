# Getting Started

## Installation
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
!!! tip
    For PyTorch installation, please refer to [Get Started](https://pytorch.org/get-started/locally/) and ensure that the right version is installed! 
    We will also keep providing support for the latest PyTorch version.
