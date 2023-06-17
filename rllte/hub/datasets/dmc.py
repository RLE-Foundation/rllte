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
import pandas as pd
from huggingface_hub import hf_hub_download


class DeepMindControl:
    """Scores and learning cures of various RL algorithms on the full
    DeepMind Control Suite benchmark.
    """

    def __init__(self) -> None:
        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", repo_type="dataset", filename="dm_control.json", subfolder="datasets"
        )
        self.dm_control_data = pd.read_json(file)

    def load_scores(self) -> Dict[str, np.ndarray]:
        """Returns final performance"""
        scores_dict = dict()
        for algo in self.dm_control_data.keys():
            scores_dict[algo] = np.array([value for _, value in self.dm_control_data[algo].items()]).T

        return scores_dict

    def load_curves(self) -> None:
        pass
