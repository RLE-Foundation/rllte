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
from rllte.agent import SAC, DrQv2
from rllte.env import make_dmc_env
from rllte.common.prototype import BaseAgent

# cheetah_run quadruped_walk quadruped_run walker_walk walker_run hopper_hop arcobot_swingup cup_catch 
# cartpole_balance cartpole_balance_sparse cartpole_swingup cartpole_swingup_sparse finger_spin finger_turn_easy
# finger_turn_hard fish_swim fish_upright hopper_stand pendulum_swingup quadruped_run reacher_easy reacher_hard swimmer_swimmer6 swimmer_swimmer15

class DMControl(Bucket):
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

        self.sup_env = ['acrobot_swingup', 'cartpole_balance', 'cartpole_balance_sparse', 
                        'cartpole_swingup', 'cartpole_swingup_sparse', 'cheetah_run', 
                        'cup_catch', 'finger_spin', 'finger_turn_easy', 
                        'finger_turn_hard', 'fish_swim', 'fish_upright', 
                        'hopper_hop', 'hopper_stand', 'pendulum_swingup',
                        'quadruped_run', 'quadruped_walk', 'reacher_easy',
                        'reacher_hard', 'swimmer_swimmer6', 'swimmer_swimmer15', 
                        'walker_run', 'walker_walk', 'walker_stand',
                        'humanoid_walk', 'humanoid_run', 'humanoid_stand'
                        ]
        self.sup_algo = ['sac']
    
    def get_obs_type(self, agent: str) -> str:
        """Returns the observation type of the agent.
        
        Args:
            agent (str): Agent name.
        
        Returns:
            Observation type.
        """
        obs_type = 'state' if agent in ['sac'] else 'pixel'
        return obs_type

    def load_scores(self, env_id: str, agent: str) -> Dict[str, np.ndarray]:
        """Returns final performance.
        
        Args:
            env_id (str): Environment ID.
            agent_id (str): Agent name.
        
        Returns:
            Test scores data array with shape (N_SEEDS, N_POINTS).
        """
        self.is_available(env_id=env_id, agent=agent.lower())

        obs_type = self.get_obs_type(agent=agent.lower())
        scores_file = f'{agent.lower()}_dmc_{obs_type}_{env_id}_scores.npy'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=scores_file, 
            subfolder="dmc/scores"
        )

        return np.load(file)

    def load_curves(self, env_id: str, agent: str) -> Dict[str, np.ndarray]:
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
        self.is_available(env_id=env_id, agent=agent.lower())

        obs_type = self.get_obs_type(agent=agent.lower())
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

        obs_type = self.get_obs_type(agent=agent.lower())
        model_file = f"{agent.lower()}_dmc_{obs_type}_{env_id}_seed_{seed}.pth"
        subfolder = f"dmc/{agent}"
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
        if agent.lower() == "sac":
            # The following hyperparameters are from the repository:
            # https://github.com/denisyarats/pytorch_sac
            envs = make_dmc_env(env_id=env_id, num_envs=1, device=device, seed=seed, from_pixels=False, visualize_reward=True)
            eval_envs = make_dmc_env(
                env_id=env_id, num_envs=1, device=device, seed=seed, from_pixels=False, visualize_reward=True
            )

            api = SAC(
                env=envs,
                eval_env=eval_envs,
                tag=f"sac_dmc_state_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_init_steps=5000,
                storage_size=10000000,
                feature_dim=50,
                batch_size=1024,
                lr=1e-4,
                eps=1e-8,
                hidden_dim=1024,
                critic_target_tau=0.005,
                actor_update_freq=1,
                critic_target_update_freq=2,
                init_fn="orthogonal",
            )
        elif agent.lower() == "drq-v2":
            # The following hyperparameters are from the repository:
            # https://github.com/facebookresearch/drqv2
            envs = make_dmc_env(
                env_id=env_id,
                num_envs=1,
                device=device,
                seed=seed,
                from_pixels=True,
                visualize_reward=False,
                frame_stack=3,
                action_repeat=2,
                asynchronous=False
            )
            eval_envs = make_dmc_env(
                env_id=env_id,
                num_envs=1,
                device=device,
                seed=seed,
                from_pixels=True,
                visualize_reward=False,
                frame_stack=3,
                action_repeat=2,
                asynchronous=False
            )
            # create agent
            api = DrQv2( # type: ignore[assignment]
                env=envs,
                eval_env=eval_envs,
                tag=f"drqv2_dmc_pixel_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                feature_dim=50,
                batch_size=256,
                lr=0.0001,
                eps=1e-8,
                hidden_dim=1024,
                critic_target_tau=0.01,
                update_every_steps=2,
                init_fn="orthogonal",
            )
        else:
            raise NotImplementedError(
                f"Agent {agent} is not supported currently, available agents are: [SAC, DDPG, TD3, DrQv2]."
            )

        return api