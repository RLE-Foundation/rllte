import flax.linen as nn
import jax.numpy as jnp

from hsuanwu.common.typing import *
from hsuanwu.xploit.encoders import CnnEncoder

class Critic(nn.Module):
    """
    Critic network.

    :param action_space: The action space of the environment.
    :param feature_dim: Number of features extracted.
    :param hidden_dim: The size of the hidden layers.
    """
    action_shape: Tuple[int]
    feature_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs, action) -> jnp.array:
        h = CnnEncoder(name='encoder')(obs)

        h = nn.Dense(self.feature_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.tanh(h)

        ''' concatenate h and action '''
        h_action = jnp.concatenate([h, action], axis=1)

        q1 = nn.Dense(self.hidden_dim)(h_action)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)

        q2 = nn.Dense(self.hidden_dim)(h_action)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)

        return q1, q2

def update_critic(
    actor: TrainState, critic: TrainState, target_critic: TrainState, 
    batch: Batch, discount: float) -> Tuple[TrainState, InfoDict]:
    next_actions = actor(batch.next_observations)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations, batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, 