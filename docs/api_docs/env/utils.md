#


## HsuanwuEnvWrapper
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/utils.py/#L11)
```python 
HsuanwuEnvWrapper(
   env: VectorEnv, device: str
)
```


---
Env wrapper for adapting to Hsuanwu engine and outputting torch tensors.


**Args**

* **env** (VectorEnv) : The vectorized environments.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

HsuanwuEnvWrapper instance.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/utils.py/#L53)
```python
.reset(
   seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None
)
```

---
Reset all parallel environments and return a batch of initial observations and info.


**Args**

* **seed** (int) : The environment reset seeds.
* **options** (Optional[dict]) : If to return the options.


**Returns**

A batch of observations and info from the vectorized environment.

### .step
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/utils.py/#L71)
```python
.step(
   actions: th.Tensor
)
```

---
Take an action for each parallel environment.


**Args**

* **actions** (Tensor) : element of :attr:`action_space` Batch of actions.


**Returns**

Batch of (observations, rewards, terminations, truncations, infos)
