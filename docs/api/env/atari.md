#


### make_atari_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/atari/__init__.py/#L63)
```python
.make_atari_env(
   env_id: str = 'Alien-v5', num_envs: int = 8, seed: int = 0, frame_stack: int = 4,
   device: torch.device = 'cuda'
)
```

---
Build Atari environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of parallel environments.
* **seed** (int) : Random seed.
* **frame_stack** (int) : Number of stacked frames.
* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

Environments instance.
