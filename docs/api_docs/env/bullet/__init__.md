#


### make_bullet_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/bullet/__init__.py/#L46)
```python
.make_bullet_env(
   env_id: str = 'AntBulletEnv-v0', num_envs: int = 1, device: str = 'cpu', seed: int = 0
)
```

---
Build PyBullet robotics environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of parallel environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.


**Returns**

Environments instance.
