#


## PPOLearner
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppo.py\#L56)
```python 
PPOLearner(
   observation_space: Dict, action_space: Dict, device: Device, feature_dim: int,
   lr: float, eps: float, hidden_dim: int, clip_range: float, n_epochs: int,
   num_mini_batch: int, vf_coef: float, ent_coef: float, aug_coef: float,
   max_grad_norm: float
)
```


---
Proximal Policy Optimization (PPO) Learner.
When 'augmentation' module is invoked, this learner will transform into Data Regularized Actor-Critic (DrAC) Learner.


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
* **n_epochs** (int) : Times of updating the policy.
* **num_mini_batch** (int) : Number of mini-batches.
* **vf_coef** (float) : Weighting coefficient of value loss.
* **ent_coef** (float) : Weighting coefficient of entropy bonus.
* **aug_coef** (float) : Weighting coefficient of augmentation loss.
* **max_grad_norm** (float) : Maximum norm of gradients.



**Returns**

PPO learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppo.py\#L126)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppo.py\#L140)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/ppo.py\#L152)
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
