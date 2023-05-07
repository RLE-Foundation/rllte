#


### make_multibinary_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/multibinary/__init__.py/#L65)
```python
.make_multibinary_env(
   env_id: str = 'multibinary_state', num_envs: int = 1, device: str = 'cpu', seed: int = 0
)
```

---
Build environments with `MultiBinary` action space for testing.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of parallel environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.


**Returns**

Environments instance.
