#


## DAAC
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L67)
```python 
DAAC(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str, feature_dim: int, lr: float, eps: float,
   hidden_dim: int, clip_range: float, policy_epochs: int, value_freq: int,
   value_epochs: int, num_mini_batch: int, vf_coef: float, ent_coef: float,
   aug_coef: float, adv_coef: float, max_grad_norm: float
)
```


---
Decoupled Advantage Actor-Critic (DAAC) agent.
When 'augmentation' module is invoked, this learner will transform into Data Regularized Decoupled Actor-Critic (DrAAC) agent.
Based on: https://github.com/rraileanu/idaac


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **clip_range** (float) : Clipping parameter.
* **policy_epochs** (int) : Times of updating the policy network.
* **value_freq** (int) : Update frequency of the value network.
* **value_epochs** (int) : Times of updating the value network.
* **num_mini_batch** (int) : Number of mini-batches.
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **aug_coef** (float) : Weighting coefficient of augmentation loss.
* **adv_ceof** (float) : Weighting coefficient of advantage loss.
* **max_grad_norm** (float) : Maximum norm of gradients.



**Returns**

DAAC learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L143)
```python
.train(
   training: bool = True
)
```

---
Set the train mode.


**Args**

* **training** (bool) : True (training) or False (testing).


**Returns**

None.

### .integrate
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L155)
```python
.integrate(
   **kwargs
)
```

---
Integrate agent and other modules (encoder, reward, ...) together

### .get_value
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L179)
```python
.get_value(
   obs: th.Tensor
)
```

---
Get estimated values for observations.


**Args**

* **obs** (Tensor) : Observations.


**Returns**

Estimated values.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L190)
```python
.act(
   obs: th.Tensor, training: bool = True, step: int = 0
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

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L213)
```python
.update(
   rollout_storage: Storage, episode: int = 0
)
```

---
Update the learner.


**Args**

* **rollout_storage** (Storage) : Hsuanwu rollout storage.
* **episode** (int) : Global training episode.


**Returns**

Training metrics such as actor loss, critic_loss, etc.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L350)
```python
.save(
   path: Path
)
```

---
Save models.


**Args**

* **path** (Path) : Storage path.


**Returns**

None.

### .load
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/daac.py/#L365)
```python
.load(
   path: str
)
```

---
Load initial parameters.


**Args**

* **path** (str) : Import path.


**Returns**

None.
