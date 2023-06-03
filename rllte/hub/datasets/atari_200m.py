from typing import Dict
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

class Atari200M(object):
    def __init__(self) -> None:
        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub",
            repo_type="dataset",
            filename="atari_200_iters_normalized_scores.npy",
            subfolder="datasets"
        )

        with open(file, 'rb') as f:
            atari_200m_scores = np.load(f, allow_pickle=True)
            atari_200m_scores = atari_200m_scores.tolist()
            
        for key, val in atari_200m_scores.items():
            atari_200m_scores[key] = np.transpose(val, axes=(1, 2, 0))
        self.atari_200_iters_normalized_scores = atari_200m_scores

    def load_scores(self) -> Dict[str, np.ndarray]:
        """Returns final performance."""

    def load_curves(self) -> np.ndarray:
        """Returns training curves."""
        return self.atari_200_iters_normalized_scores