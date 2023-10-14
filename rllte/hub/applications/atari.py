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


import os

from rllte.agent import A2C, IMPALA, PPO
from rllte.env import make_atari_env, make_envpool_atari_env

os.environ["OMP_NUM_THREADS"] = "1"


class Atari:
    """Train an agent on the Atari games.
        Environment link: https://github.com/Farama-Foundation/Arcade-Learning-Environment
        Added algorithms: [PPO, A2C, IMPALA]

    Args:
        agent (str): The agent to train.
        env_id (str): The environment name.
        seed (int): The random seed.
        device (str): The device to train on.
        envpool (bool): Whether to use envpool or not.

    Returns:
        Training applications.
    """

    def __init__(self, agent: str = "PPO", env_id: str = "Pong-v5", seed: int = 1, device: str = "cuda", envpool: bool = False) -> None:
        if agent == "PPO":
            # The following hyperparameters are from the repository:
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
            if envpool:
                # Since the asynchronous mode achieved much lower training performance than the synchronous mode, 
                # we recommend using the synchronous mode currently.
                envs = make_envpool_atari_env(env_id=env_id, num_envs=8, device=device, seed=seed, asynchronous=False)
                eval_envs = make_envpool_atari_env(env_id=env_id, num_envs=8, device=device, seed=seed, asynchronous=False)
            else:
                envs = make_atari_env(env_id=env_id, num_envs=8, device=device, seed=seed)
                eval_envs = make_atari_env(env_id=env_id, num_envs=8, device=device, seed=seed, asynchronous=False)

            self.agent = PPO(
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
            self.anneal_lr = True
        elif agent == "A2C":
            # The following hyperparameters are from the repository:
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
            if envpool:
                envs = make_envpool_atari_env(env_id=env_id, num_envs=16, device=device, seed=seed)
                eval_envs = make_envpool_atari_env(env_id=env_id, num_envs=16, device=device, seed=seed, asynchronous=False)
            else:
                envs = make_atari_env(env_id=env_id, num_envs=16, device=device, seed=seed)
                eval_envs = make_atari_env(env_id=env_id, num_envs=16, device=device, seed=seed, asynchronous=False)

            self.agent = A2C( # type: ignore[assignment]
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
            self.anneal_lr = False
        elif agent == "IMPALA":
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

    def train(
        self,
        num_train_steps: int = 50000000,
        log_interval: int = 1,
        eval_interval: int = 100,
        save_interval: int = 100,
        num_eval_episodes: int = 10,
        th_compile: bool = False
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
            anneal_lr=self.anneal_lr
        )
