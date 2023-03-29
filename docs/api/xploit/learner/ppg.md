#


## PPGLearner
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/ppg.py/#L115)
```python 
PPGLearner(
   observation_space: Space, action_space: Space, action_type: str,
   device: torch.device = 'cuda', feature_dim: int = 256, lr: float = 0.0005,
   eps: float = 1e-05, hidden_dim: int = 256, clip_range: float = 0.2,
   num_policy_mini_batch: int = 8, num_aux_mini_batch: int = 4, vf_coef: float = 0.5,
   ent_coef: float = 0.01, max_grad_norm: float = 0.5, policy_epochs: int = 32,
   aux_epochs: int = 6, kl_coef: float = 1.0, num_aux_grad_accum: int = 1
)
```


---
Phasic Policy Gradient (PPG) Learner.


**Args**

* **observation_space**  : Observation space of the environment.
* **action_space**  : Action space of the environment.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim**  : Number of features extracted.
* **lr**  : The learning rate.
* **eps**  : Term added to the denominator to improve numerical stability.
* **hidden_dim**  : The size of the hidden layers.
* **clip_range**  : Clipping parameter.
* **num_policy_mini_batch**  : Number of mini-batches in policy phase.
* **num_aux_mini_batch**  : Number of mini-batches in auxiliary phase.
* **vf_coef**  : Weighting coefficient of value loss.
* **ent_coef**  : Weighting coefficient of entropy bonus.
* **max_grad_norm**  : Maximum norm of gradients.
* **policy_epochs**  : Number of iterations in the policy phase.
* **aux_epochs**  : Number of iterations in the auxiliary phase.
* **kl_coef**  : Weighting coefficient of divergence loss.
* **num_aux_grad_accum**  : Number of gradient accumulation for auxiliary phase update.



**Returns**

PPG learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/ppg.py/#L192)
```python
.train(
   training = True
)
```

---
Set the train mode.


**Args**

* **training**  : True (training) or False (testing).


**Returns**

None.

### .set_dist
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/ppg.py/#L207)
```python
.set_dist(
   dist: Distribution
)
```

---
Set the distribution for actor.


**Args**

* **dist**  : Hsuanwu distribution class.


**Returns**

None.

### .get_value
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/ppg.py/#L220)
```python
.get_value(
   obs: Tensor
)
```

---
Get estimated values for observations.


**Args**

* **obs**  : Observations.


**Returns**

Estimated values.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/ppg.py/#L233)
```python
.act(
   obs: Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions.


**Args**

* **obs**  : Observation tensor.
* **training**  : Training or testing.
* **step**  : Global training step.


**Returns**

Sampled actions.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/ppg.py/#L254)
```python
.update(
   rollout_buffer: Any, episode: int = 0
)
```

---
Update learner.


**Args**

* **rollout_buffer**  : Hsuanwu rollout buffer.
* **episode**  : Global training episode.


**Returns**

Training metrics such as actor loss, critic_loss, etc.
