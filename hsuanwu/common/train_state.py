from flax import struct

import jax.numpy as jnp
import flax.linen as nn
import optax
import flax
import jax
import os

from hsuanwu.common.typing import *

@struct.dataclass
class TrainState:
    """
    Create train state for jax-based modules.

    :param step: Counter starts at 0 and is incremented by every call to `.apply_gradients()`.
    :param apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    :param tx: An Optax gradient transformation.
    :param opt_state: The state for `tx`.
    """

    step: int
    apply_fn: Callable[..., Any] = struct.field(pytree_node=False)
    tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.array],
        tx: Optional[optax.GradientTransformation] = None) -> 'TrainState':

        variables = model_def.init(*inputs)
        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1, apply_fn=model_def.apply, params=params, tx=tx, opt_state=opt_state)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.apply_fn({'params': self.params}, *args, **kwargs)
    
    def apply_gradients(
        self, 
        loss_fn: Optional[Callable[[Params], Any]] = None,
        grads: Optional[Any] = None,
        has_aux: bool = True) -> Union[Tuple['TrainState', Any], 'TrainState']:

        if grads is None:
            grad_fn = jax.grad(loss_fn, has_aux=has_aux)
            if has_aux:
                grads, aux = grad_fn(self.params)
            else:
                grads = grad_fn(self.params)
        else:
            assert (has_aux, 'When grads are provided, expects no aux outputs.')
        
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)

        if has_aux:
            return new_model, aux
        else:
            return new_model
    
    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'TrainState':
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)