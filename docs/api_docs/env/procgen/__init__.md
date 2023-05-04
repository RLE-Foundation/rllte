#


### make_procgen_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/procgen/__init__.py/#L44)
```python
.make_procgen_env(
   env_id: str = 'bigfish', num_envs: int = 64, gamma: float = 0.99, num_levels: int = 200,
   start_level: int = 0, distribution_mode: str = 'easy', device: str = 'cpu'
)
```

---
Build Prcogen environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of parallel environments.
* **gamma** (float) : A discount factor.
* **num_levels** (int) : The number of unique levels that can be generated.
    Set to 0 to use unlimited levels.
* **start_level** (int) : The lowest seed that will be used to generated levels.
    'start_level' and 'num_levels' fully specify the set of possible levels.
* **distribution_mode** (str) : What variant of the levels to use, the options are "easy",
    "hard", "extreme", "memory", "exploration".
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

Environments instance.
