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
from rllte.agent import PPO, PPG, DAAC
from rllte.xploit.encoder import EspeholtResidualEncoder
from rllte.env import make_envpool_procgen_env
from rllte.common.prototype import BaseAgent


class Procgen(Bucket):
    """Scores and learning cures of various RL algorithms on the full Procgen benchmark.
    Environment link: https://github.com/openai/procgen
    Number of environments: 16
    Number of training steps: 25,000,000
    Number of seeds: 10
    Added algorithms: [PPO]
    """
    def __init__(self) -> None:
        super().__init__()

        self.sup_env = ['bigfish', 'bossfight', 'caveflyer', 'chaser',
                        'climber', 'coinrun', 'dodgeball', 'fruitbot',
                        'heist', 'jumper', 'leaper', 'maze',
                        'miner', 'ninja', 'plunder', 'starpilot'
                        ]
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

        scores_file = f'{agent.lower()}_procgen_{env_id}_scores.npy'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=scores_file, 
            subfolder="procgen/scores"
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

        curves_file = f'{agent.lower()}_procgen_{env_id}_curves.npz'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=curves_file,
            subfolder="procgen/curves"
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
        
        model_file = f"{agent.lower()}_procgen_{env_id}_seed_{seed}.pth"
        subfolder = f"procgen/{agent}"
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
        envs = make_envpool_procgen_env(
                env_id=env_id,
                num_envs=64,
                device=device,
                seed=seed,
                gamma=0.99,
                num_levels=200,
                start_level=0,
                distribution_mode="easy",
                asynchronous=False
            )
        eval_envs = make_envpool_procgen_env(
                env_id=env_id,
                num_envs=1,
                device=device,
                seed=seed,
                gamma=0.99,
                num_levels=0,
                start_level=0,
                distribution_mode="easy",
                asynchronous=False,
            )

        feature_dim = 256
        if agent.lower() == "ppo":
            # The following hyperparameters are from the repository:
            # https://github.com/rraileanu/auto-drac
            api = PPO(
                env=envs,
                eval_env=eval_envs,
                tag=f"ppo_procgen_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=256,
                feature_dim=256,
                batch_size=2048,
                lr=5e-4,
                eps=1e-5,
                clip_range=0.2,
                clip_range_vf=0.2,
                n_epochs=3,
                vf_coef=0.5,
                ent_coef=0.01,
                max_grad_norm=0.5,
                init_fn="xavier_uniform",
            )
        elif agent.lower() == "daac":
            # Best hyperparameters for DAAC reported in
            # https://github.com/rraileanu/idaac/blob/main/hyperparams.py
            if env_id in ['plunder', 'chaser']:
                value_epochs = 1
            else:
                value_epochs = 9
            
            if env_id in ['miner', 'bigfish', 'dodgeball']:
                value_freq = 32
            elif env_id == 'plunder':
                value_freq = 8
            else:
                value_freq = 1
            
            if env_id == 'plunder':
                adv_coef = 0.3
            elif env_id == 'chaser':
                adv_coef = 0.15
            elif env_id in ['climber', 'bigfish']:
                adv_coef = 0.05
            else:
                adv_coef = 0.25

            api = DAAC( # type: ignore[assignment]
                env=envs,
                eval_env=eval_envs,
                tag=f"daac_procgen_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=256,
                feature_dim=feature_dim,
                batch_size=2048,
                lr=5e-4,
                eps=1e-5,
                clip_range=0.2,
                clip_range_vf=0.2,
                policy_epochs=1,
                value_epochs=value_epochs,
                value_freq=value_freq,
                vf_coef=0.5,
                ent_coef=0.01,
                adv_coef=adv_coef,
                max_grad_norm=0.5,
                init_fn="xavier_uniform",
            )
        else:
            raise NotImplementedError(f"Agent {agent} is not supported currently, available agents are: [PPO, DAAC, PPG].")

        # set the residual encoder
        encoder = EspeholtResidualEncoder(observation_space=envs.observation_space, feature_dim=feature_dim)
        api.set(encoder=encoder)

        return api