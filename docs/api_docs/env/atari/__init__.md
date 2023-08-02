#


### make_atari_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/atari/__init__.py/#L74)
```python
.make_atari_env(
   env_id: str = 'Alien-v5', num_envs: int = 8, device: str = 'cpu', seed: int = 1,
   frame_stack: int = 4, parallel: bool = True
)
```

---
Build Atari environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.
* **frame_stack** (int) : Number of stacked frames.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`. 
    For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.


**Returns**

The vectorized environment.

----


### make_envpool_atari_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/atari/__init__.py/#L45)
```python
.make_envpool_atari_env(
   env_id: str = 'Alien-v5', num_envs: int = 8, device: str = 'cpu', seed: int = 1,
   parallel: bool = True
)
```

---
Build Atari environments with `envpool`.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`. 
    For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.

