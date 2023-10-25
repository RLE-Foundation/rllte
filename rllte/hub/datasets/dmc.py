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


class DMControl(BaseDataset):
    """Scores and learning cures of various RL algorithms on the full
        DeepMind Control Suite benchmark.
    Environment link: https://github.com/google-deepmind/dm_control
    Number of environments: 27
    Number of training steps: 10,000,000 for humanoid, 2,000,000 for others
    Number of seeds: 10
    Added algorithms: [SAC, DrQ-v2]
    """
    def __init__(self) -> None:
        super().__init__()

        self.sup_env = ['cheetah_run']
        self.sup_algo = ['sac']
        self.sup_level = ['random', 'expert']
        self.sup_obs_type = ['state', 'pixel']

    def is_available(self, type: str, env_id: str, agent: Optional[str] = None, level: Optional[str] = None, obs_type: Optional[str] = None) -> None:
        """Returns True if the dataset is available, False otherwise."""

        if type == "scores":
            assert env_id in self.sup_env and agent in self.sup_algo, \
                f"Scores for `{env_id}` and `{agent}` are not available currently!"
        elif type == "curves":
            assert env_id in self.sup_env and agent in self.sup_algo, \
                f"Curves for `{env_id}` and `{agent}` are not available currently!"
        elif type == "demonstrations":
            assert env_id in self.sup_env and level in self.sup_level and obs_type in self.sup_obs_type, \
                f"Demonstrations for `{env_id}` and level `{level}` are not available currently!"
        else:
            raise NotImplementedError

    def load_scores(self, env_id: str, agent: str, obs_type: str = 'state') -> Dict[str, np.ndarray]:
        """Returns final performance.
        
        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
            obs_type (str): A type from ['state', 'pixel'].
        
        Returns:
            Test scores data array with shape (N_SEEDS, N_POINTS).
        """
        self.is_available(type="scores", env_id=env_id, agent=agent.lower())

        scores_file = f'{agent.lower()}_dmc_{obs_type}_{env_id}_scores.npy'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=scores_file, 
            subfolder="dmc/scores"
        )

        return np.load(file)

    def load_curves(self, env_id: str, agent: str, obs_type: str = 'state') -> Dict[str, np.ndarray]:
        """Returns learning curves using a `Dict` of NumPy arrays.

        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
            obs_type (str): A type from ['state', 'pixel'].
        
        Returns:
            Learning curves data with structure:
            curves
            ├── train: np.ndarray(shape=(N_SEEDS, N_POINTS))
            └── eval:  np.ndarray(shape=(N_SEEDS, N_POINTS))
        """
        self.is_available(type="curves", env_id=env_id, agent=agent.lower())

        curves_file = f'{agent.lower()}_dmc_{obs_type}_{env_id}_curves.npz'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=curves_file,
            subfolder="dmc/curves"
        )

        curves_dict = np.load(file, allow_pickle=True)
        curves_dict = dict(curves_dict)

        return curves_dict

    def load_demonstrations(self, env_id: str, level: str, obs_type: str = 'state') -> Dict[str, np.ndarray]:
        """Returns demonstrations using a `Dict` of NumPy arrays.

        Args:
            env_id (str): Environment ID.
            level (str): A level from ['expert', 'random', 'exploration'].
            obs_type (str): A type from ['state', 'pixel'].
        
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
        assert obs_type in self.sup_obs_type, f"Observation type `{obs_type}` is not supported!"
        self.is_available(type="demonstrations", env_id=env_id, level=level, obs_type=obs_type)

        demons_file = f'{env_id}_{level}_{obs_type}_demonstrations.npz'
        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub",
            repo_type="model",
            filename=demons_file,
            subfolder="dmc/demonstrations",
        )

        demonstrations_dict = np.load(file, allow_pickle=True)
        demonstrations_dict = {key: value.item() for key, value in demonstrations_dict.items()}

        return demonstrations_dict
