#


### make_minigrid_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/minigrid/__init__.py/#L88)
```python
.make_minigrid_env(
   env_id: str = 'MiniGrid-DoorKey-5x5-v0', num_envs: int = 8,
   fully_observable: bool = True, fully_numerical: bool = False, seed: int = 0,
   frame_stack: int = 1, device: str = 'cpu', parallel: bool = True
)
```

---
Build MiniGrid environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **fully_observable** (bool) : Fully observable gridworld using a compact grid encoding instead of the agent view.
* **fully_numerical** (bool) : Transforms the observation space (that has a textual component) to a fully numerical 
    observation space, where the textual instructions are replaced by arrays representing the indices of each 
    word in a fixed vocabulary.
* **seed** (int) : Random seed.
* **frame_stack** (int) : Number of stacked frames.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`. 
    For `Distributed` algorithms, in which `SyncVectorEnv` is required
    and reward clip will be used before environment vectorization.


**Returns**

The vectorized environment.
