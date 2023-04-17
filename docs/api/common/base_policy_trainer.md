#


## BasePolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L52)
```python 
BasePolicyTrainer(
   cfgs: DictConfig, train_env: Env, test_env: Env = None
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


### .global_step
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L103)
```python
.global_step()
```

---
Get global training steps.

### .global_episode
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L108)
```python
.global_episode()
```

---
Get global training episodes.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L311)
```python
.act(
   obs: Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions based on observations.


**Args**

* **obs**  : Observations.
* **training**  : training mode, True or False.
* **step**  : Global training step.


**Returns**

Sampled actions.

### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L324)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L328)
```python
.test()
```

---
Testing function.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/base_policy_trainer.py\#L332)
```python
.save()
```

---
Save the trained model.
