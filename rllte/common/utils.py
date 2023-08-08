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


import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch as th
from torch import nn


class ExportModel(nn.Module):
    """Module for model export.

    Args:
        encoder (nn.Module): Encoder network.
        actor (nn.Module): Actor network.

    Returns:
        Export model format.
    """

    def __init__(self, encoder: nn.Module, actor: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.actor = actor

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Only for model inference.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Deterministic actions.
        """
        return self.actor(self.encoder(obs))


class eval_mode:
    """Set the evaluation mode.

    Args:
        models (nn.Module): Models.

    Returns:
        None.
    """

    def __init__(self, *models) -> None:
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.mode(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.mode(state)
        return False


def to_numpy(xs: Tuple[th.Tensor, ...]) -> Tuple[np.ndarray, ...]:
    """Converts torch tensors to numpy arrays.

    Args:
        xs (Tuple[th.Tensor, ...]): Torch tensors.

    Returns:
        Numpy arrays.
    """
    for x in xs:
        print(x.size())
    return tuple(x[0].cpu().numpy() for x in xs)


def pretty_json(hp: Dict) -> str:
    """Returns a pretty json string.

    Args:
        hp (Dict): Hyperparameters.

    Returns:
        Pretty json string.
    """
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


def get_episode_statistics(infos: Dict) -> Tuple[List, List]:
    """Get the episode statistics.

    Args:
        infos (Dict): Information.

    Returns:
        Episode rewards and lengths.
    """
    indices = np.nonzero(infos["episode"]["l"])

    return infos["episode"]["r"][indices].tolist(), infos["episode"]["l"][indices].tolist()


def get_npu_name() -> str:
    """Get NPU name."""
    str_command = "npu-smi info"
    out = os.popen(str_command)
    text_content = out.read()
    out.close()
    lines = text_content.split("\n")
    npu_name_line = lines[6]
    name_part = npu_name_line.split("|")[1]
    npu_name = name_part.split()[-1]

    return npu_name
