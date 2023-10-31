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


from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
from torch import nn
import numpy as np

from rllte.common.prototype import BaseAgent

class Bucket(ABC):
    """Bucket class for storing scores, learning curves, and models."""
    def __init__(self) -> None:
        super().__init__()

        self.sup_env: List = []
        self.sup_algo: List = []


    def is_available(self, env_id: str, agent: Optional[str] = None) -> None:
        """Check if the dataset is available."""
        assert env_id in self.sup_env and agent in self.sup_algo, \
            f"Datasets for `{env_id}` and `{agent}` are not available currently!"


    @abstractmethod
    def load_scores(self, env_id: str, agent: str) -> np.ndarray:
        """Returns final performance.
        
        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
        
        Returns:
            Test scores data array with shape (N_SEEDS, N_POINTS).
        """
        

    @abstractmethod
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
    
    @abstractmethod
    def load_models(self, 
                    env_id: str, 
                    agent: str, 
                    seed: int, 
                    device: str = "cpu"
                    ) -> nn.Module:
        """Load the model from the hub.

        Args:
            env_id (str): Environment ID.
            agent (str): Agent name.
            seed (int): The seed to load.
            device (str): The device to load the model on.

        Returns:
            The loaded model.
        """

    @abstractmethod
    def load_apis(self, 
                  env_id: str, 
                  agent: str, 
                  seed: int, 
                  device: str = "cpu"
                  ) -> BaseAgent:
        """Load the a training API.

        Args:
            env_id (str): Environment ID.
            agent (str): Agent name.
            seed (int): The seed to load.
            device (str): The device to load the model on.

        Returns:
            The loaded API.
        """