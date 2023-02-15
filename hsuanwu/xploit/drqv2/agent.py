import jax.numpy as jnp
import functools
import optax
import jax


from hsuanwu.common.typing import *

class DrQv2Agent:
    """
    Learner for continuous control tasks.
    Current learner: DrQ-v2
    Paper: Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning
    Link: https://openreview.net/pdf?id=_SJ-_yyes8

    :param obs_space: The observation shape of the environment.
    :param action_shape: The action shape of the environment.
    :param feature_dim: Number of features extracted.
    :param hidden_dim: The size of the hidden layers.
    :param lr: The learning rate.
    :param critic_target_tau: The critic Q-function soft-update rate.
    :param update_every_steps: The agent update frequency.
    :param num_expl_steps: The exploration steps.
    :param stddev_schedule: The exploration std schedule.
    :param stddev_clip: The exploration std clip range.
    """
    def __init__(self,
                obs_shape: Tuple[int] = (84, 84, 3), 
                device: torch.device = 'cuda',
                action_shape: Tuple[int] = (7, ),
                feature_dim: int = 50,
                hidden_dim: int = 1024,
                lr: float = 1e-4,
                critic_target_tau: float = 0.01,
                num_expl_steps: int = 2000,
                update_every_steps: int = 2,
                stddev_schedule: str = 'linear(1.0, 0.1, 100000)',
                stddev_clip: float = 0.3) -> None:
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

    def act(self, obs, training=True):
        pass
    
    def update(self, batch: Batch):
        pass
