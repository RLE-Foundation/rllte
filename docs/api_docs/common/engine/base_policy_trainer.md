#


## BasePolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L50)
```python 
BasePolicyTrainer(
   cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None
)
```


---
Base class of policy trainer.


**Args**

* **cfgs** (DictConfig) : Dict config for configuring RL algorithms.
* **train_env** (Env) : A Gym-like environment for training.
* **test_env** (Env) : A Gym-like environment for testing.


**Returns**

Base policy trainer instance.


**Methods:**


### .get_npu_name
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L111)
```python
.get_npu_name()
```

---
Get NPU name.

### .global_step
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L125)
```python
.global_step()
```

---
Get global training steps.

### .global_episode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L130)
```python
.global_episode()
```

---
Get global training episodes.

### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L318)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L322)
```python
.test()
```

---
Testing function.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L326)
```python
.save()
```

---
Save the trained model.
