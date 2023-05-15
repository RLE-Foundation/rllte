#


### make_robosuite_env
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/env/robosuite/__init__.py\#L47)
```python
.make_robosuite_env(
   env_id: str = 'Lift_Panda', num_envs: int = 1, device: str = 'cpu', seed: int = 0,
   distributed: bool = False, has_renderer: bool = False,
   has_offscreen_renderer: bool = False, use_camera_obs: bool = False
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
* **has_renderer** (bool) : If true, render the simulation state in
    a viewer instead of headless mode.
* **has_offscreen_renderer** (bool) : True if using off-screen rendering.
* **use_camera_obs** (bool) : True for using image observations.


**Returns**

Environments instance.
