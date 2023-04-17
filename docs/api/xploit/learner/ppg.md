#


## PPGLearner
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppg.py\#L60)
```python 
PPGLearner(
   observation_space: Dict, action_space: Dict, device: Device,
   feature_dim: int = 256, lr: float = 0.0005, eps: float = 1e-05, hidden_dim: int = 256,
   clip_range: float = 0.2, num_policy_mini_batch: int = 8, num_aux_mini_batch: int = 4,
   vf_coef: float = 0.5, ent_coef: float = 0.01, max_grad_norm: float = 0.5,
   policy_epochs: int = 32, aux_epochs: int = 6, kl_coef: float = 1.0,
   num_aux_grad_accum: int = 1
)
```


---
Phasic Policy Gradient (PPG) Learner.


**Args**

* **observation_space** (Dict) : Observation space of the environment.
    For supporting Hydra, the original 'observation_space' is transformed into a dict like {"shape": observation_space.shape, }.
* **action_space** (Dict) : Action shape of the environment.
    For supporting Hydra, the original 'action_space' is transformed into a dict like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **clip_range** (float) : Clipping parameter.
* **num_policy_mini_batch** (int) : Number of mini-batches in policy phase.
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **max_grad_norm** (float) : Maximum norm of gradients.
* **policy_epochs** (int) : Number of iterations in the policy phase.
* **aux_epochs** (int) : Number of iterations in the auxiliary phase.
* **kl_coef** (float) : Weighting coefficient of divergence loss.
* **num_aux_grad_accum** (int) : Number of gradient accumulation for auxiliary phase update.

num_aux_mini_batch (int) Number of mini-batches in auxiliary phase.


**Returns**

PPG learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppg.py\#L144)
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

### .get_value
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppg.py\#L158)
```python
.get_value(
   obs: Tensor
)
```

---
Get estimated values for observations.


**Args**

* **obs** (Tensor) : Observations.


**Returns**

Estimated values.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppg.py\#L170)
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
