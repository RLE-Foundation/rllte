import jax.numpy as jnp
import jax
from gym import spaces
import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from hsuanwu.xploit.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(
    buffer_size=500000,
    observation_space=spaces.Box(low=0., high=1., shape=[84, 84, 3]),
    action_space=spaces.Box(low=0., high=1., shape=[10, ]),
    n_envs=1
)