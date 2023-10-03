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


from rllte.agent import A2C, PPO
from rllte.env import make_minigrid_env


class MiniGrid:
    """Train an agent on the Atari games.
        Environment link: https://github.com/Farama-Foundation/Arcade-Learning-Environment
        Added algorithms: [PPO, A2C, IMPALA]

    Args:
        agent (str): The agent to train.
        env_id (str): The environment name.
        seed (int): The random seed.
        device (str): The device to train on.

    Returns:
        Training applications.
    """

    def __init__(
        self, agent: str = "PPO", env_id: str = "MiniGrid-DoorKey-5x5-v0", seed: int = 1, device: str = "cuda"
    ) -> None:
        if agent == "PPO":
            # The following hyperparameters are from the repository:
            # https://github.com/lcswillems/rl-starter-files
            envs = make_minigrid_env(env_id=env_id, num_envs=8, device=device, seed=seed)
            eval_envs = make_minigrid_env(env_id=env_id, num_envs=1, device=device, seed=seed)

            self.agent = PPO(
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
        elif agent == "A2C":
            # The following hyperparameters are from the repository:
            # https://github.com/lcswillems/rl-starter-files
            envs = make_minigrid_env(env_id=env_id, num_envs=1, device=device, seed=seed)
            eval_envs = make_minigrid_env(env_id=env_id, num_envs=1, device=device, seed=seed)
            self.agent = A2C( # type: ignore[assignment]
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
            raise NotImplementedError(f"Agent {agent} is not supported currently, available agents are: [A2C, PPO, IMPALA].")

    def train(
        self,
        num_train_steps: int = 1000000,
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
