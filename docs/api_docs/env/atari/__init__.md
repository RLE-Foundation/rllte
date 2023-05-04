#


### make_atari_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/atari/__init__.py/#L12)
```python
.make_atari_env(
   env_id: str = 'Alien-v5', num_envs: int = 8, device: str = 'cpu', seed: int = 1,
   frame_stack: int = 4
)
```

---
Build Atari environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of parallel environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.
* **frame_stack** (int) : Number of stacked frames.


**Returns**

Environments instance.
