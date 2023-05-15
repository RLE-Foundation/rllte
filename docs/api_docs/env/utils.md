#


## HsuanwuEnvWrapper
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/env/utils.py\#L12)
```python 
HsuanwuEnvWrapper(
   env_fn: Callable, num_envs: int = 1, device: str = 'cpu', parallel: bool = True
)
```


---
Env wrapper for adapting to Hsuanwu engine and outputting torch tensors.


**Args**

* **env_fn** (Callable) : Function that creates the environments.
* **num_envs** (int) : Number of environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.


**Returns**

HsuanwuEnvWrapper instance.
