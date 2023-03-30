#


## OffPolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L10)
```python 
OffPolicyTrainer(
   train_env: Env, test_env: Env, cfgs: DictConfig
)
```


---
Trainer for off-policy algorithms.


**Args**

* **train_env**  : A Gym-like environment for training.
* **test_env**  : A Gym-like environment for testing.
* **cfgs**  : Dict config for configuring RL algorithms.


**Returns**

Off-policy trainer instance.


**Methods:**


### .replay_iter
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L60)
```python
.replay_iter()
```

---
Create iterable dataloader.

### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L66)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/off_policy_trainer.py/#L122)
```python
.test()
```

---
Testing function.
