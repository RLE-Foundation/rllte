import jax.numpy as jnp
import functools
import optax
import jax


from hsuanwu.common.typing import *
from hsuanwu.common.train_state import TrainState
from hsuanwu.xploit.drqv2.actor import Actor, update_actor
from hsuanwu.xploit.drqv2.critic import Critic, update_critic


def target_update(critic: TrainState, target_critic: TrainState, tau: float) -> TrainState:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    actor: TrainState, critic: TrainState, target_critic: TrainState,
    batch: Batch, discount: float, tau: float, update_target: bool
    ) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, InfoDict]:

    new_critic, critic_info = update_critic(actor, critic, target_critic, batch, discount)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic
    
    new_actor, actor_info = update_actor(actor, new_critic, batch)

    return new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info}

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

        # init models
        self.rng = jax.random.PRNGKey(0)
        self.rng, actor_key, critic_key = jax.random.split(self.rng, 3)
        init_obs = jnp.ones((1, *obs_shape))
        init_actions = jnp.ones((1, action_shape[0]))

        # actor train state
        self.actor = TrainState.create(
            model_def=Actor(action_shape=action_shape, feature_dim=feature_dim, hidden_dim=hidden_dim),
            inputs=[actor_key, init_obs],
            tx=optax.adam(learning_rate=lr))
        # critic train state
        self.critic = TrainState.create(
            model_def=Critic(action_shape=action_shape, feature_dim=feature_dim, hidden_dim=hidden_dim),
            inputs=[critic_key, init_obs, init_actions],
            tx=optax.adam(learning_rate=lr))
        # target critic
        self.target_critic = TrainState.create(
            model_def=Critic(action_shape=action_shape, feature_dim=feature_dim, hidden_dim=hidden_dim),
            inputs=[critic_key, init_obs, init_actions])
        
        self.step = 0
    
    def act(self, obs, training=True):
        action = self.actor(jnp.expand_dims(obs, 0))
        if not training:
            return action
        
        self.rng, key = jax.random.split(self.rng)
        if self.step < self.num_expl_steps:
            action = jax.random.uniform(key=key, shape=action.shape, minval=-1.0, maxval=1.0)
        else:
            action = action + jax.random.normal(key=key, shape=action.shape)
        
        return action
    
    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        new_actor, new_critic, new_target_critic, info = _update_jit(
            self.actor, self.critic, self.target_critic, batch, 0.99,
            self.critic_target_tau, self.step % self.update_every_steps == 0)

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info
