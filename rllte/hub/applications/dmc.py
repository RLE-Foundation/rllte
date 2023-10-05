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


from rllte.agent import SAC, DrQv2
from rllte.env import make_dmc_env


class DMControl:
    """Train an agent on the DeepMind Control Suite.
        Environment link: https://github.com/google-deepmind/dm_control
        Added algorithms: [SAC, DDPG, TD3, DrQ-v2 (Pixel)]

    Args:
        agent (str): The agent to train.
        env_id (str): The environment name.
        seed (int): The random seed.
        device (str): The device to train on.

    Returns:
        Training applications.
    """

    def __init__(self, agent: str = "SAC", env_id: str = "humanoid_run", seed: int = 1, device: str = "cuda") -> None:
        if agent == "SAC":
            # The following hyperparameters are from the repository:
            # https://github.com/denisyarats/pytorch_sac
            envs = make_dmc_env(env_id=env_id, num_envs=1, device=device, seed=seed, from_pixels=False, visualize_reward=True)
            eval_envs = make_dmc_env(
                env_id=env_id, num_envs=1, device=device, seed=seed, from_pixels=False, visualize_reward=True
            )

            self.agent = SAC(
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
        elif agent == "DrQ-v2":
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
            self.agent = DrQv2( # type: ignore[assignment]
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

    def train(
        self,
        num_train_steps: int = 10000000,
        log_interval: int = 1000,
        eval_interval: int = 10000,
        save_interval: int = 10000,
        num_eval_episodes: int = 10,
        th_compile: bool = False,
    ) -> None:
        """Training function.

        Args:
            num_train_steps (int): The number of training steps.
            log_interval (int): The interval (in steps) of logging.
            eval_interval (int): The interval (in steps) of evaluation.
            save_interval (int): The interval (in steps) of saving model.
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
