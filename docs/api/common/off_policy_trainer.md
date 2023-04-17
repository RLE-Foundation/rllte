#


## OffPolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/off_policy_trainer.py\#L11)
```python 
OffPolicyTrainer(
   cfgs: DictConfig, train_env: Env, test_env: Env = None
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


### .replay_iter
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/off_policy_trainer.py\#L78)
```python
.replay_iter()
```

---
Create iterable dataloader.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/off_policy_trainer.py\#L84)
```python
.act(
   obs: Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions based on observations.


**Args**

* **obs** (Tensor) : Observations.
* **training** (bool) : training mode, True or False.
* **step** (int) : Global training step.


**Returns**

Sampled actions.

### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/off_policy_trainer.py\#L107)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/off_policy_trainer.py\#L176)
```python
.test()
```

---
Testing function.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/off_policy_trainer.py\#L204)
```python
.save()
```

---
Save the trained model.
