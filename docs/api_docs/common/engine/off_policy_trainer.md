#


## OffPolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L16)
```python 
OffPolicyTrainer(
   cfgs: omegaconf.DictConfig, train_env: gym.Env, test_env: gym.Env = None
)
```


---
Trainer for off-policy algorithms.


**Args**

* **cfgs** (DictConfig) : Dict config for configuring RL algorithms.
* **train_env** (Env) : A Gym-like environment for training.
* **test_env** (Env) : A Gym-like environment for testing.


**Returns**

Off-policy trainer instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L64)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L130)
```python
.test()
```

---
Testing function.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L158)
```python
.save()
```

---
Save the trained model.
