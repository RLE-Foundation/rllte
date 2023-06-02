#


### make_bullet_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/bullet/__init__.py/#L47)
```python
.make_bullet_env(
   env_id: str = 'AntBulletEnv-v0', num_envs: int = 1, device: str = 'cpu', seed: int = 0,
   distributed: bool = False
)
```

---
Build PyBullet robotics environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.
* **distributed** (bool) : For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.


**Returns**

Environments instance.
