#


### make_bullet_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/bullet/__init__.py/#L81)
```python
.make_bullet_env(
   env_id: str = 'AntBulletEnv-v0', num_envs: int = 1, device: str = 'cpu', seed: int = 0,
   parallel: bool = True
)
```

---
Create PyBullet robotics environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device to convert the data.
* **seed** (int) : Random seed.
* **parallel** (bool) : `True` for creating asynchronous environments, and `False`
    for creating synchronous environments.


**Returns**

The vectorized environments.
