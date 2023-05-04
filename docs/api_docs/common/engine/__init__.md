#


## HsuanwuEngine
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/__init__.py/#L12)
```python 
HsuanwuEngine(
   cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None
)
```


---
Hsuanwu RL engine.


**Args**

* **cfgs** (DictConfig) : Dict config for configuring RL algorithms.
* **train_env** (Env) : A Gym-like environment for training.
* **test_env** (Env) : A Gym-like environment for testing.


**Returns**

Hsuanwu engine instance.


**Methods:**


### .invoke
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/__init__.py/#L39)
```python
.invoke()
```

---
Invoke the engine to perform training.
