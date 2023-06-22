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


from typing import Tuple

import numpy as np
import torch as th
from torch import nn


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Soft update of the target network.

    Args:
        net (nn.Module): Network to update.
        target_net (nn.Module): Target network.
        tau (float): Interpolation parameter.

    Returns:
        None
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def to_torch(xs: Tuple[np.ndarray, ...], device: th.device) -> Tuple[th.Tensor, ...]:
    """Convert numpy arrays to torch tensors.

    Args:
        xs (Tuple[np.ndarray, ...]): Numpy arrays.
        device (th.device): Device to store the tensors.

    Returns:
        Tuple[th.Tensor, ...]: Torch tensors.
    """
    return tuple(th.as_tensor(x, device=device).float() for x in xs)
