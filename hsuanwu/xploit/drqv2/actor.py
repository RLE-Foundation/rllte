import flax.linen as nn
import jax.numpy as jnp

from hsuanwu.common.typing import *
from hsuanwu.common.train_state import TrainState
from hsuanwu.xploit.encoders import CnnEncoder

class Actor(nn.Module):
    """
    Actor network.

    :param action_shape: The action shape of the environment.
    :param feature_dim: Number of features extracted.
    :param hidden_dim: The size of the hidden layers.
    """
    action_shape: Tuple[int]
    feature_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs) -> jnp.ndarray:
        h = CnnEncoder(name='encoder')(obs)
        # detach
        h = jax.lax.stop_gradient(h)

        h = nn.Dense(self.feature_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.tanh(h)
        ''' policy '''
        x = nn.Dense(self.hidden_dim)(h)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_shape[0])(x)
        x = nn.tanh(x)
        return x

def update_actor(
    actor: TrainState, critic: TrainState, 
    batch: Batch) -> Tuple[TrainState, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = actor.apply_fn({'params': actor_params}, batch.observations)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = - q.mean()
        return actor_loss, {'actor_loss': actor_loss}

    new_actor, info = actor.apply_gradients(actor_loss_fn)

    return new_actor, info