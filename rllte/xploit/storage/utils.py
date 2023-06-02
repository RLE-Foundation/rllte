import io
import random
from pathlib import Path
from typing import Dict

import numpy as np


def dump_episode(episode: Dict, fn: Path) -> None:
    """Save episode as *.npz file.

    Args:
        episode: episode to be save.
        fn: file path.

    Return:
        None.
    """
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn: Path):
    """Load episode from *.npz file.

    Args:
        fn: file path.

    Return:
        Episode data.
    """
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def episode_len(episode: Dict) -> int:
    """Get the length of an episode.

    Args:
        episode: Selected episode.

    Returns:
        episode length.
    """
    return len(episode["observation"]) - 1


def worker_init_fn(worker_id):
    """Function for dataloader initialization.

    Args:
        workder_id: .

    Returns:
        None.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)
