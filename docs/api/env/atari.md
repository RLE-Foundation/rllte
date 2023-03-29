#


### make_atari_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/atari/__init__.py/#L18)
```python
.make_atari_env(
   env_id: str = 'Alien-v5', num_envs: int = 8, seed: int = 0, frame_stack: int = 4,
   device: torch.device = 'cuda'
)
```

---
Build Atari environments.


**Args**

* **env_id**  : Name of environment.
* **num_envs**  : Number of parallel environments.
* **seed**  : Random seed.
* **frame_stack**  : Number of stacked frames.
* **device**  : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

Environments instance.
