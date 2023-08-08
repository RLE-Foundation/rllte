# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import Callable

from torch import nn


def _identity(m):
    """Identity initialization."""
    pass


def _orthogonal(m):
    """Orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def _xavier_uniform(m):
    """Xavier uniform initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def _xavier_normal(m):
    """Xavier normal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def get_init_fn(method: str = "orthogonal") -> Callable:  # : c901
    """Returns a network initialization function.

    Args:
        method (str): Initialization method name.

    Returns:
        Initialization function.
    """
    if method == "orthogonal":
        return _orthogonal
    elif method == "xavier_normal":
        return _xavier_normal
    elif method == "xavier_uniform":
        return _xavier_uniform
    else:
        return _identity
