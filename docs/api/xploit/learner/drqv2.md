#


## DrQv2Learner
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/drqv2.py\#L60)
```python 
DrQv2Learner(
   observation_space: Dict, action_space: Dict, device: Device, feature_dim: int,
   lr: float, eps: float, hidden_dim: int, critic_target_tau: float,
   update_every_steps: int
)
```


---
Data Regularized-Q v2 (DrQ-v2).
When 'augmentation' module is deprecated, this learner will transform into Deep Deterministic Policy Gradient (DDPG) Learner.


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
* **critic_target_tau**  : The critic Q-function soft-update rate.
* **update_every_steps** (int) : The agent update frequency.



**Returns**

DrQv2 learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/drqv2.py\#L119)
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

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/drqv2.py\#L134)
```python
.update(
   replay_iter: Iterable, step: int = 0
)
```

---
Update the learner.


**Args**

* **replay_iter** (Iterable) : Hsuanwu replay storage iterable dataloader.
* **step** (int) : Global training step.


**Returns**

Training metrics such as actor loss, critic_loss, etc.

### .update_critic
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/drqv2.py\#L193)
```python
.update_critic(
   obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs: Tensor,
   step: int
)
```

---
Update the critic network.


**Args**

* **obs** (Tensor) : Observations.
* **action** (Tensor) : Actions.
* **reward** (Tensor) : Rewards.
* **discount** (Tensor) : discounts.
* **next_obs** (Tensor) : Next observations.
* **step** (int) : Global training step.


**Returns**

Critic loss metrics.

### .update_actor
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/drqv2.py\#L242)
```python
.update_actor(
   obs: Tensor, step: int
)
```

---
Update the actor network.


**Args**

* **obs** (Tensor) : Observations.
* **step** (int) : Global training step.


**Returns**

Actor loss metrics.
