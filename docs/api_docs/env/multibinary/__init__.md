#


### make_multibinary_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/multibinary/__init__.py/#L118)
```python
.make_multibinary_env(
   env_id: str = 'MultiBinary-State', num_envs: int = 1, device: str = 'cpu', seed: int = 0,
   parallel: bool = True
)
```

---
Build environments with `MultiBinary` action space for testing.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device to convert the data.
* **seed** (int) : Random seed.
* **parallel** (bool) : `True` for creating asynchronous environments, and `False`
    for creating synchronous environments.


**Returns**

The vectorized environments.
