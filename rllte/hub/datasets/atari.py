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


from typing import Dict

import numpy as np
from huggingface_hub import hf_hub_download


class Atari:
    """Scores and learning cures of various RL algorithms on the full Atari benchmark.
    Environment link: https://github.com/Farama-Foundation/Arcade-Learning-Environment
    Number of environments: 57
    Number of training steps: 50,000,000
    Number of seeds: 10
    Added algorithms: [PPO]
    """

    def __init__(self) -> None:
        pass

    def load_scores(self) -> Dict[str, np.ndarray]:
        """Returns final performance."""

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", repo_type="dataset", filename="atari_scores.npy", subfolder="atari"
        )

        scores_dict = np.load(file, allow_pickle=True).item()

        return scores_dict

    def load_curves(self) -> Dict[str, np.ndarray]:
        """Returns learning curves using a `Dict` of NumPy arrays:
        curves = {
            "ppo": {
                "train": {"Pong-v5": np.ndarray(shape=(Number of seeds, Number of points)), ...},
                "eval": {"Pong-v5": np.ndarray(shape=(Number of seeds, Number of points)), ...},
            },
            ...
        }
        """

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", repo_type="dataset", filename="atari_curves.npy", subfolder="atari"
        )

        curves_dict = np.load(file, allow_pickle=True).item()

        return curves_dict
