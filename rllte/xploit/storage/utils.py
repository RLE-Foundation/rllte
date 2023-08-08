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


import io
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch as th


def episode_len(episode: Dict[str, np.ndarray]) -> int:
    """Returns the length of an episode.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        episode (Dict[str, np.ndarray]): Selected episode.

    Returns:
        Episode length.
    """
    return next(iter(episode.values())).shape[0]


def save_episode(episode: Dict[str, np.ndarray], fn: Path) -> None:
    """Saves an episode to a `.npz` file.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        episode (Dict[str, np.ndarray]): Episode to be saved.
        fn (Path): File path.

    Returns:
        None.
    """
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn: Path) -> Dict[str, np.ndarray]:
    """Loads an episode from a `.npz` file.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        fn (Path): File path.

    Returns:
        Episode data.
    """
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def worker_init_fn(worker_id: int) -> None:
    """Sets the random seed for each worker.
        Borrowed from: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py

    Args:
        worker_id (int): Worker ID.

    Returns:
        None.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def to_torch(xs: Tuple[np.ndarray, ...], device: th.device) -> Tuple[th.Tensor, ...]:
    """Convert numpy arrays to torch tensors.

    Args:
        xs (Tuple[np.ndarray, ...]): Numpy arrays.
        device (th.device): Device to store the tensors.

    Returns:
        Tuple[th.Tensor, ...]: Torch tensors.
    """
    return tuple(th.as_tensor(x, device=device).float() for x in xs)
