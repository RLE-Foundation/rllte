#


### make_atari_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/atari/__init__.py/#L37)
```python
.make_atari_env(
   env_id: str = 'Alien-v5', num_envs: int = 8, device: str = 'cpu', seed: int = 1,
   frame_stack: int = 4, distributed: bool = False
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
* **distributed** (bool) : For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.


**Returns**

The vectorized environment.
