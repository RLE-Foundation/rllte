#


### make_rllte_env
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/env/utils.py/#L34)
```python
.make_rllte_env(
   env_id: Union[str, Callable[..., gym.Env]], num_envs: int = 1, seed: int = 1,
   device: str = 'cpu', parallel: bool = True, env_kwargs: Optional[Dict[str,
   Any]] = None
)
```

---
Create environments that adapt to rllte engine.


**Args**

* **env_id** (Union[str, Callable[..., gym.Env]]) : either the env ID, the env class or a callable returning an env
* **num_envs** (int) : Number of environments.
* **seed** (int) : Random seed.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **parallel** (bool) : `True` for `AsyncVectorEnv` and `False` for `SyncVectorEnv`.
* **env_kwargs**  : Optional keyword argument to pass to the env constructor


**Returns**

Environment wrapped by `TorchVecEnvWrapper`.
