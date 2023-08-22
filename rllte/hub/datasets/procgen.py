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


class Procgen:
    """Scores and learning cures of various RL algorithms on the full Procgen benchmark. 
        Environment link: https://github.com/openai/procgen
        Number of environments: 16
        Number of training steps: 25,000,000
        Number of seeds: 10
        Added algorithms: [PPO]
    """

    def __init__(self) -> None:
        pass

    def load_scores(self) -> Dict[str, np.ndarray]:
        """Returns final performance."""

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", repo_type="dataset", filename="procgen_scores.npy", subfolder="procgen"
        )

        scores_dict = np.load(file, allow_pickle=True).item()

        return scores_dict

    def load_curves(self) -> Dict[str, np.ndarray]:
        """Returns learning curves using a `Dict` of arrays:
            curves = {
                "ppo": {
                    "train": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
                    "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
                },
                "daac": {
                    "train": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
                    "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
                },
                ...
            }
        """

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", repo_type="dataset", filename="procgen_curves.npy", subfolder="procgen"
        )

        curves_dict = np.load(file, allow_pickle=True).item()

        return curves_dict
