#


## VecEnvWrapper
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L12)
```python 
RllteEnvWrapper(
   env_fn: Callable, num_envs: int = 1, device: str = 'cpu', parallel: bool = True
)
```


---
Env wrapper for adapting to rllte engine and outputting torch tensors.


**Args**

* **env_fn** (Callable) : Function that creates the environments.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.


**Returns**

RllteEnvWrapper instance.

----


## TorchVecEnvWrapper
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L43)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L63)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L81)
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
