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
from typing import Dict, Callable
from torch import nn

import numpy as np
import torch as th

from rllte.hub.bucket import Bucket
from rllte.agent import A2C, PPO, IMPALA
from rllte.env import make_atari_env, make_envpool_atari_env
from rllte.common.prototype import BaseAgent


class Atari(Bucket):
    """Scores and learning cures of various RL algorithms on the full Atari benchmark.
    Environment link: https://github.com/Farama-Foundation/Arcade-Learning-Environment
    Number of environments: 57
    Number of training steps: 10,000,000
    Number of seeds: 10
    Added algorithms: [PPO]
    """
    def __init__(self) -> None:
        super().__init__()

        self.sup_env = ['Alien-v5', 'Amidar-v5', 'Assault-v5', 'Asterix-v5', 'Asteroids-v5', 'Atlantis-v5', 'YarsRevenge-v5',
                        'BankHeist-v5', 'BattleZone-v5', 'BeamRider-v5', 'Berzerk-v5', 'Bowling-v5', 'Boxing-v5', 'Breakout-v5',
                        'Centipede-v5', 'ChopperCommand-v5', 'CrazyClimber-v5', 'Defender-v5', 'DemonAttack-v5', 'DoubleDunk-v5', 'Zaxxon-v5',
                        'Enduro-v5', 'FishingDerby-v5', 'Freeway-v5', 'Frostbite-v5', 'Gopher-v5', 'Gravitar-v5', 'Hero-v5',
                        'IceHockey-v5', 'Jamesbond-v5', 'Kangaroo-v5', 'Krull-v5', 'KungFuMaster-v5', 'MontezumaRevenge-v5', 'Pitfall-v5',
                        'PrivateEye-v5', 'Qbert-v5', 'Riverraid-v5', 'RoadRunner-v5', 'Robotank-v5', 'Seaquest-v5', 'Phoenix-v5', 'Pong-v5',
                        'Skiing-v5', 'Solaris-v5', 'SpaceInvaders-v5', 'StarGunner-v5', 'Surround-v5', 'Tennis-v5', 'TimePilot-v5',
                        'Tutankham-v5', 'UpNDown-v5', 'Venture-v5', 'VideoPinball-v5', 'WizardOfWor-v5', 'MsPacman-v5', 'NameThisGame-v5'
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

        scores_file = f'{agent.lower()}_atari_{env_id}_scores.npy'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=scores_file, 
            subfolder="atari/scores"
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

        curves_file = f'{agent.lower()}_atari_{env_id}_curves.npz'

        file = hf_hub_download(
            repo_id="RLE-Foundation/rllte-hub", 
            repo_type="model", 
            filename=curves_file,
            subfolder="atari/curves"
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
        
        model_file = f"{agent.lower()}_atari_{env_id}_seed_{seed}.pth"
        subfolder = f"atari/{agent}"
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
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
            # Since the asynchronous mode achieved much lower training performance than the synchronous mode, 
            # we recommend using the synchronous mode currently.
            envs = make_envpool_atari_env(env_id=env_id, num_envs=8, device=device, seed=seed, asynchronous=False)
            eval_envs = make_envpool_atari_env(env_id=env_id, num_envs=8, device=device, seed=seed, asynchronous=False)

            api = PPO(
                env=envs,
                eval_env=eval_envs,
                tag=f"ppo_atari_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=128,
                feature_dim=512,
                batch_size=256,
                lr=2.5e-4,
                eps=1e-5,
                clip_range=0.1,
                clip_range_vf=0.1,
                n_epochs=4,
                vf_coef=0.5,
                ent_coef=0.01,
                max_grad_norm=0.5,
                discount=0.99,
                init_fn="orthogonal"
            )
        elif agent.lower() == "a2c":
            # The following hyperparameters are from the repository:
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
            envs = make_envpool_atari_env(env_id=env_id, num_envs=16, device=device, seed=seed)
            eval_envs = make_envpool_atari_env(env_id=env_id, num_envs=16, device=device, seed=seed, asynchronous=False)

            api = A2C( # type: ignore[assignment]
                env=envs,
                eval_env=eval_envs,
                tag=f"a2c_atari_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=5,
                feature_dim=512,
                hidden_dim=256,
                batch_size=80,
                lr=10e-4,
                eps=1e-5,
                n_epochs=4,
                vf_coef=0.5,
                ent_coef=0.01,
                max_grad_norm=0.5,
                init_fn="orthogonal",
            )
        elif agent.lower() == "impala":
            # The following hyperparameters are from the repository:
            # https://github.com/facebookresearch/torchbeast
            envs = make_atari_env(env_id=env_id, device=device, seed=seed, num_envs=45, asynchronous=False)
            eval_envs = make_atari_env(env_id=env_id, device=device, seed=seed, num_envs=1, asynchronous=False)
            self.agent = IMPALA( # type: ignore[assignment]
                env=envs,
                eval_env=eval_envs,
                tag=f"impala_atari_{env_id}_seed_{seed}",
                seed=seed,
                device=device,
                num_steps=80,
                num_actors=45,
                num_learners=4,
                num_storages=60,
                feature_dim=512,
            )
        else:
            raise NotImplementedError(f"Agent {agent} is not supported currently, available agents are: [A2C, PPO, IMPALA].")

        return api