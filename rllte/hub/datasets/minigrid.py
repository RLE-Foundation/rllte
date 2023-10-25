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


from huggingface_hub import hf_hub_download
from typing import Dict, Optional

import numpy as np
from rllte.hub.datasets.base import BaseDataset


class MiniGrid:
    """Scores and learning cures of various RL algorithms on the MiniGrid benchmark.
    Environment link: https://github.com/Farama-Foundation/Minigrid
    Number of environments: 16
    Number of training steps: 1,000,000
    Number of seeds: 10
    Added algorithms: [A2C]
    """
    def __init__(self) -> None:
        super().__init__()

        self.sup_env = ['Empty-6x6-v0']
        self.sup_algo = ['ppo']
        self.sup_level = ['random', 'expert']

    def is_available(self, type: str, env_id: str, agent: Optional[str] = None, level: Optional[str] = None) -> None:
        """Returns True if the dataset is available, False otherwise."""

        if type == "scores":
            assert env_id in self.sup_env and agent in self.sup_algo, \
                f"Scores for `{env_id}` and `{agent}` are not available currently!"
        elif type == "curves":
            assert env_id in self.sup_env and agent in self.sup_algo, \
                f"Curves for `{env_id}` and `{agent}` are not available currently!"
        elif type == "demonstrations":
            assert env_id in self.sup_env and level in self.sup_level, \
                f"Demonstrations for `{env_id}` and level `{level}` are not available currently!"
        else:
            raise NotImplementedError

    def load_scores(self, env_id: str, agent: str) -> np.ndarray:
        """Returns final performance.
        
        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
        
        Returns:
            Test scores data array with shape (N_SEEDS, N_POINTS).
        """
        self.is_available(type="scores", env_id=env_id, agent=agent.lower())

        scores_file = f'{agent.lower()}_minigrid_{env_id}_scores.npy'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=scores_file, 
            subfolder="minigrid/scores"
        )

        return np.load(file)

    def load_curves(self, env_id: str, agent: str) -> Dict[str, np.ndarray]:
        """Returns learning curves using a `Dict` of NumPy arrays.

        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
        
        Returns:
            Learning curves data with structure:
            curves
            ├── train: np.ndarray(shape=(N_SEEDS, N_POINTS))
            └── eval:  np.ndarray(shape=(N_SEEDS, N_POINTS))
        """
        self.is_available(type="curves", env_id=env_id, agent=agent.lower())

        curves_file = f'{agent.lower()}_minigrid_{env_id}_curves.npz'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=curves_file,
            subfolder="minigrid/curves"
        )

        curves_dict = np.load(file, allow_pickle=True)
        curves_dict = dict(curves_dict)

        return curves_dict

    def load_demonstrations(self, env_id: str, level: str) -> Dict[str, np.ndarray]:
        """Returns demonstrations using a `Dict` of NumPy arrays.

        Args:
            env_id (str): Environment ID.
            level (str): A level from ['expert', 'random', 'exploration'].
        
        Returns:
            Demonstrations data with structure:
            demonstrations
            ├── episode_0
            │   ├── observations
            │   ├── actions
            │   ├── rewards
            │   ├── terminateds
            │   └── truncateds
            ├── episode_1
            │   ├── observations
            │   ├── actions
            │   ├── rewards
            │   ├── terminateds
            │   └── truncateds
            └── ...
        """
        self.is_available(type="demonstrations", env_id=env_id, level=level)

        demons_file = f'{env_id}_{level}_demonstrations.npz'
        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub",
            repo_type="model",
            filename=demons_file,
            subfolder="minigrid/demonstrations",
        )

        demonstrations_dict = np.load(file, allow_pickle=True)
        demonstrations_dict = {key: value.item() for key, value in demonstrations_dict.items()}

        return demonstrations_dict
