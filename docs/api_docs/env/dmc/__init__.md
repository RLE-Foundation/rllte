#


### make_dmc_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/dmc/__init__.py/#L35)
```python
.make_dmc_env(
   env_id: str = 'humanoid_run', num_envs: int = 1, device: str = 'cpu', seed: int = 1,
   visualize_reward: bool = True, from_pixels: bool = False, height: int = 84,
   width: int = 84, frame_stack: int = 3, action_repeat: int = 1, asynchronous: bool = True
)
```

---
Create DeepMind Control Suite environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device to convert the data.
* **seed** (int) : Random seed.
* **visualize_reward** (bool) : Opposite to `from_pixels`.
* **from_pixels** (bool) : Provide image-based observations or not.
* **height** (int) : Image observation height.
* **width** (int) : Image observation width.
* **frame_stack** (int) : Number of stacked frames.
* **action_repeat** (int) : Number of action repeats.
* **asynchronous** (bool) : `True` for creating asynchronous environments,
    and `False` for creating synchronous environments.


**Returns**

The vectorized environments.
