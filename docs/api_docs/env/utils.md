#


## TorchVecEnvWrapper
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L66)
```python 
TorchVecEnvWrapper(
   env: VectorEnv, device: str
)
```


---
Env wrapper for outputting torch tensors.


**Args**

* **env** (VectorEnv) : The vectorized environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

TorchVecEnvWrapper instance.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L86)
```python
.reset(
   seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None
)
```

---
Reset all environments and return a batch of initial observations and info.


**Args**

* **seed** (int) : The environment reset seeds.
* **options** (Optional[dict]) : If to return the options.


**Returns**

A batch of observations and info from the vectorized environment.

### .step
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L104)
```python
.step(
   actions: th.Tensor
)
```

---
Take an action for each environment.


**Args**

* **actions** (Tensor) : element of :attr:`action_space` Batch of actions.


**Returns**

Batch of (observations, rewards, terminations, truncations, infos)

----


### make_rllte_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L12)
```python
.make_rllte_env(
   env_id: Union[str, Callable[..., gym.Env]], num_envs: int = 1, seed: int = 1,
   device: str = 'cpu', parallel: bool = True, env_kwargs: Optional[Dict[str,
   Any]] = None
)
```

---
Create environments that adapt to rllte engine.


**Args**

* **env_id** (Union[str, Callable[..., gym.Env]]) : either the env ID, the env class or a callable returning an env
* **num_envs** (int) : Number of environments.
* **seed** (int) : Random seed.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.
* **env_kwargs**  : Optional keyword argument to pass to the env constructor


**Returns**

Environment wrapped by `TorchVecEnvWrapper`.
