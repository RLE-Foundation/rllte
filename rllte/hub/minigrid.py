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
from torch import nn

import numpy as np
import torch as th
from rllte.hub.bucket import Bucket
from rllte.agent import A2C, PPO
from rllte.env import make_minigrid_env
from rllte.common.prototype import BaseAgent


class MiniGrid(Bucket):
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

    def load_scores(self, env_id: str, agent: str) -> np.ndarray:
        """Returns final performance.
        
        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
        
        Returns:
            Test scores data array with shape (N_SEEDS, N_POINTS).
        """
        self.is_available(env_id=env_id, agent=agent.lower())

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
        self.is_available(env_id=env_id, agent=agent.lower())

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
        self.is_available(env_id=env_id, agent=agent.lower())
        
        model_file = f"{agent.lower()}_minigrid_{env_id}_seed_{seed}.pth"
        subfolder = f"minigrid/{agent}"
        file = hf_hub_download(repo_id="RLE-Foundation/rllte-hub", repo_type="model", filename=model_file, subfolder=subfolder)
        model = th.load(file, map_location=device)

        return model.eval()
    

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
        if agent.lower() == "ppo":
            # The following hyperparameters are from the repository:
            # https://github.com/lcswillems/rl-starter-files
            envs = make_minigrid_env(env_id=env_id, num_envs=8, device=device, seed=seed)
            eval_envs = make_minigrid_env(env_id=env_id, num_envs=1, device=device, seed=seed)

            api = PPO(
                env=envs,
                eval_env=eval_envs,
                tag=f"ppo_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=128,
                feature_dim=256,
                batch_size=64,
                hidden_dim=64,
                lr=2.5e-4,
                eps=1e-5,
                clip_range=0.2,
                clip_range_vf=0.2,
                n_epochs=10,
                vf_coef=0.5,
                ent_coef=0.0,
                max_grad_norm=0.5,
                init_fn="orthogonal",
            )
        elif agent.lower() == "a2c":
            # The following hyperparameters are from the repository:
            # https://github.com/lcswillems/rl-starter-files
            envs = make_minigrid_env(env_id=env_id, num_envs=1, device=device, seed=seed)
            eval_envs = make_minigrid_env(env_id=env_id, num_envs=1, device=device, seed=seed)
            api = A2C( # type: ignore[assignment]
                env=envs,
                eval_env=eval_envs,
                tag=f"a2c_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=5,
                feature_dim=256,
                hidden_dim=256,
                batch_size=256,
                lr=1e-3,
                eps=1e-5,
                n_epochs=4,
                vf_coef=0.5,
                ent_coef=0.01,
                max_grad_norm=0.5,
                init_fn="orthogonal",
            )
        else:
            raise NotImplementedError(f"Agent {agent} is not supported currently, available agents are: [A2C, PPO].")

        return api