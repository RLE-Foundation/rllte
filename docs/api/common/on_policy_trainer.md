#


## OnPolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/on_policy_trainer.py/#L12)
```python 
OnPolicyTrainer(
   train_env: Env, test_env: Env, cfgs: DictConfig
)
```


---
Trainer for on-policy algorithms.


**Args**

* **train_env**  : A Gym-like environment for training.
* **test_env**  : A Gym-like environment for testing.
* **cfgs**  : Dict config for configuring RL algorithms.


**Returns**

On-policy trainer instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/on_policy_trainer.py/#L52)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/common/engine/on_policy_trainer.py/#L120)
```python
.test()
```

---
Testing function.
