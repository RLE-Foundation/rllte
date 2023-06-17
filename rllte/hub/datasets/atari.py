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


import numpy as np
from huggingface_hub import hf_hub_download


class Atari:
    def __init__(self) -> None:
        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub",
            repo_type="dataset",
            filename="atari_200_iters_normalized_scores.npy",
            subfolder="datasets",
        )

        with open(file, "rb") as f:
            atari_200m_scores = np.load(f, allow_pickle=True)
            atari_200m_scores = atari_200m_scores.tolist()

        for key, val in atari_200m_scores.items():
            atari_200m_scores[key] = np.transpose(val, axes=(1, 2, 0))
        self.atari_200_iters_normalized_scores = atari_200m_scores

    def load_scores(self) -> None:
        """Returns final performance."""

    def load_curves(self) -> np.ndarray:
        """Returns training curves."""
        return self.atari_200_iters_normalized_scores
