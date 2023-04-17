#


## DrACLearner
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drac.py/#L85)
```python 
DrACLearner(
   observation_space: Space, action_space: Space, action_type: str,
   device: torch.device = 'cuda', feature_dim: int = 256, lr: float = 0.0005,
   eps: float = 1e-05, hidden_dim: int = 256, clip_range: float = 0.2, n_epochs: int = 3,
   num_mini_batch: int = 8, vf_coef: float = 0.5, ent_coef: float = 0.01,
   aug_coef: float = 0.1, max_grad_norm: float = 0.5
)
```


---
Data Regularized Actor-Critic (DrAC) Learner.


**Args**

* **observation_space**  : Observation space of the environment.
* **action_space**  : Action shape of the environment.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim**  : Number of features extracted.
* **lr**  : The learning rate.
* **eps**  : Term added to the denominator to improve numerical stability.
* **hidden_dim**  : The size of the hidden layers.
* **clip_range**  : Clipping parameter.
* **n_epochs**  : Times of updating the policy.
* **num_mini_batch**  : Number of mini-batches.
* **vf_coef**  : Weighting coefficient of value loss.
* **ent_coef**  : Weighting coefficient of entropy bonus.
* **aug_coef**  : Weighting coefficient of augmentation loss.
* **max_grad_norm**  : Maximum norm of gradients.



**Returns**

DrAC learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drac.py/#L150)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drac.py/#L164)
```python
.set_dist(
   dist
)
```

---
Set the distribution for actor.


**Args**

* **dist**  : Hsuanwu distribution class.


**Returns**

None.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drac.py/#L176)
```python
.act(
   obs: ndarray, training: bool = True, step: int = 0
)
```

---
Make actions based on observations.


**Args**

* **obs**  : Observations.
* **training**  : training mode, True or False.
* **step**  : Global training step.


**Returns**

Sampled actions.

### .get_value
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drac.py/#L200)
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

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drac.py/#L212)
```python
.update(
   rollout_buffer: Any, episode: int = 0
)
```

---
Update the learner.


**Args**

* **rollout_buffer**  : Hsuanwu rollout buffer.
* **episode**  : Global training episode.


**Returns**

Training metrics such as actor loss, critic_loss, etc.
