#


## PPG
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L66)
```python 
PPG(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str, feature_dim: int = 256, lr: float = 0.0005,
   eps: float = 1e-05, hidden_dim: int = 256, clip_range: float = 0.2,
   num_policy_mini_batch: int = 8, num_aux_mini_batch: int = 4, vf_coef: float = 0.5,
   ent_coef: float = 0.01, aug_coef: float = 0.1, max_grad_norm: float = 0.5,
   policy_epochs: int = 32, aux_epochs: int = 6, kl_coef: float = 1.0,
   num_aux_grad_accum: int = 1
)
```


---
Phasic Policy Gradient (PPG) agent.
Based on: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppg_procgen.py


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
* **num_policy_mini_batch** (int) : Number of mini-batches in policy phase.
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **aug_coef** (float) : Weighting coefficient of augmentation loss.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **policy_epochs** (int) : Number of iterations in the policy phase.
* **aux_epochs** (int) : Number of iterations in the auxiliary phase.
* **kl_coef** (float) : Weighting coefficient of divergence loss.
* **num_aux_grad_accum** (int) : Number of gradient accumulation for auxiliary phase update.

num_aux_mini_batch (int) Number of mini-batches in auxiliary phase.


**Returns**

PPG agent instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L151)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L163)
```python
.integrate(
   **kwargs
)
```

---
Integrate agent and other modules (encoder, reward, ...) together

### .get_value
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L178)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L189)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L211)
```python
.update(
   rollout_storage: Storage, episode: int = 0
)
```

---
Update the agent.


**Args**

* **rollout_storage** (Storage) : Hsuanwu rollout storage.
* **episode** (int) : Global training episode.


**Returns**

Training metrics such as actor loss, critic_loss, etc.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L412)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/ppg.py/#L427)
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
