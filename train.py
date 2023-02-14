import jax.numpy as jnp
import jax
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from hsuanwu.xploit.drqv2.agent import DrQv2Agent
from hsuanwu.common.typing import Batch
from flax.training import train_state
import time

agent = DrQv2Agent()
obs = jnp.ones(shape=(1, 84, 84, 3))

agent.step = 5000
action = agent.act(obs=obs)
print(action)

s = time.perf_counter()
for i in range(10000):
    batch = Batch(
        observations=jnp.ones((128, 84, 84, 3)),
        actions=jnp.ones((128, 7)),
        rewards=jnp.ones((128, 1)),
        masks=jnp.zeros((128, 1)),
        next_observations=jnp.ones((128, 84, 84, 3))
    )
    info = agent.update(batch)
    print(info['actor_loss'])
e = time.perf_counter()
print(e - s)