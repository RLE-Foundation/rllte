#


### make_multibinary_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/multibinary/__init__.py/#L118)
```python
.make_multibinary_env(
   env_id: str = 'multibinary_state', num_envs: int = 1, device: str = 'cpu', seed: int = 0,
   parallel: bool = True
)
```

---
Build environments with `MultiBinary` action space for testing.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`. 
    For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.


**Returns**

The vectorized environment.
