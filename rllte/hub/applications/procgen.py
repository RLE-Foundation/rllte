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


from rllte.agent import DAAC, PPO
from rllte.env import make_envpool_procgen_env
from rllte.xploit.encoder import EspeholtResidualEncoder


class Procgen:
    """Train an agent on the Atari games.
        Environment link: https://github.com/openai/procgen
        Added algorithms: [PPO, DAAC, PPG]

    Args:
        agent (str): The agent to train.
        env_id (str): The environment name.
        seed (int): The random seed.
        device (str): The device to train on.

    Returns:
        Training applications.
    """

    def __init__(self, agent: str = "PPO", env_id: str = "bigfish", seed: int = 1, device: str = "cuda") -> None:
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
        if agent == "PPO":
            # The following hyperparameters are from the repository:
            # https://github.com/rraileanu/auto-drac
            self.agent = PPO(
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
        elif agent == "DAAC":
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

            self.agent = DAAC( # type: ignore[assignment]
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
        self.agent.set(encoder=encoder)

    def train(
        self,
        num_train_steps: int = 25000000,
        log_interval: int = 1,
        eval_interval: int = 100,
        save_interval: int = 100,
        num_eval_episodes: int = 10,
        th_compile: bool = False,
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            log_interval (int): The interval (in episodes) of logging.
            eval_interval (int): The interval (in episodes) of evaluation.
            save_interval (int): The interval (in episodes) of saving model.
            num_eval_episodes (int): The number of evaluation episodes.
            th_compile (bool): Whether to use `th.compile` or not.

        Returns:
            None.
        """
        self.agent.train(
            num_train_steps=num_train_steps,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            num_eval_episodes=num_eval_episodes,
            th_compile=th_compile,
        )
