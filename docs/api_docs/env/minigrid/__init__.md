#


### make_minigrid_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/minigrid/__init__.py/#L90)
```python
.make_minigrid_env(
   env_id: str = 'MiniGrid-DoorKey-5x5-v0', num_envs: int = 8,
   fully_observable: bool = True, fully_numerical: bool = False, seed: int = 0,
   frame_stack: int = 1, device: str = 'cpu', asynchronous: bool = True
)
```

---
Create MiniGrid environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of environments.
* **fully_observable** (bool) : Fully observable gridworld using a compact grid encoding instead of the agent view.
* **fully_numerical** (bool) : Transforms the observation space (that has a textual component) to a fully numerical
    observation space, where the textual instructions are replaced by arrays representing the indices of each
    word in a fixed vocabulary.
* **seed** (int) : Random seed.
* **frame_stack** (int) : Number of stacked frames.
* **device** (str) : Device to convert the data.
* **asynchronous** (bool) : `True` for creating asynchronous environments,
    and `False` for creating synchronous environments.


**Returns**

The vectorized environments.
