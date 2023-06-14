import io
import random
from pathlib import Path
from typing import Dict

import numpy as np


def episode_len(episode: Dict[str, np.ndarray]) -> int:
    """Returns the length of an episode.
    
    Args:
        episode (Dict[str, np.ndarray]): Selected episode.

    Returns:
        Episode length.
    """
    return next(iter(episode.values())).shape[0]

def save_episode(episode: Dict[str, np.ndarray], fn: Path) -> None:
    """Saves an episode to a `.npz` file.

    Args:
        episode (Dict[str, np.ndarray]): Episode to be saved.
        fn (Path): File path.

    Returns:
        None.
    """
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())

def load_episode(fn: Path) -> Dict[str, np.ndarray]:
    """Loads an episode from a `.npz` file.

    Args:
        fn (Path): File path.

    Returns:
        Episode data.
    """
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

def worker_init_fn(worker_id: int) -> None:
    """Sets the random seed for each worker.

    Args:
        worker_id (int): Worker ID.
    
    Returns:
        None.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)