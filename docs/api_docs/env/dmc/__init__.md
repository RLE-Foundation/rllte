#


### make_dmc_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/dmc/__init__.py/#L39)
```python
.make_dmc_env(
   env_id: str = 'cartpole_balance', num_envs: int = 1, device: str = 'cpu', seed: int = 1,
   visualize_reward: bool = False, from_pixels: bool = True, height: int = 84,
   width: int = 84, frame_stack: int = 3, action_repeat: int = 2
)
```

---
Build DeepMind Control Suite environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **seed** (int) : Random seed.
* **visualize_reward** (bool) : True when 'from_pixels' is False, False when 'from_pixels' is True.
* **from_pixels** (bool) : Provide image-based observations or not.
* **height** (int) : Image observation height.
* **width** (int) : Image observation width.
* **frame_stack** (int) : Number of stacked frames.
* **action_repeat** (int) : Number of action repeats.


**Returns**

The vectorized environment.
