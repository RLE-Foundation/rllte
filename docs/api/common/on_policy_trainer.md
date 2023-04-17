#


## OnPolicyTrainer
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/on_policy_trainer.py\#L12)
```python 
OnPolicyTrainer(
   cfgs: DictConfig, train_env: Env, test_env: Env = None
)
```


---
Trainer for on-policy algorithms.


**Args**

* **cfgs** (DictConfig) : Dict config for configuring RL algorithms.
* **train_env** (Env) : A Gym-like environment for training.
* **test_env** (Env) : A Gym-like environment for testing.


**Returns**

On-policy trainer instance.


**Methods:**


### .act
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/on_policy_trainer.py\#L62)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/on_policy_trainer.py\#L84)
```python
.train()
```

---
Training function.

### .test
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/on_policy_trainer.py\#L167)
```python
.test()
```

---
Testing function.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/common/engine/on_policy_trainer.py\#L191)
```python
.save()
```

---
Save the trained model.
