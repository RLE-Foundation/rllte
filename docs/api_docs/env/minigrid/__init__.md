#


### make_minigrid_env
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/env/minigrid/__init__.py\#L26)
```python
.make_minigrid_env(
   env_id: str = 'MiniGrid-DoorKey-5x5-v0', num_envs: int = 8,
   fully_observable: bool = True, seed: int = 0, frame_stack: int = 1, device: str = 'cpu',
   distributed: bool = False
)
```

---
Build MiniGrid environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **fully_observable** (bool) : 'True' for using fully observable RGB image as observation.
* **seed** (int) : Random seed.
* **frame_stack** (int) : Number of stacked frames.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **distributed** (bool) : For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.


**Returns**

Environments instance.
