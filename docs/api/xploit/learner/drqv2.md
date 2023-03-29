#


## DrQv2Learner
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L111)
```python 
DrQv2Learner(
   observation_space: Space, action_space: Space, action_type: str,
   device: torch.device = 'cuda', feature_dim: int = 50, lr: float = 0.0001,
   eps: float = 8e-05, hidden_dim: int = 1024, critic_target_tau: float = 0.01,
   num_init_steps: int = 2000, update_every_steps: int = 2,
   stddev_schedule: str = 'linear(1.0, 0.1, 100000)', stddev_clip: float = 0.3
)
```


---
Data Regularized-Q v2 (DrQ-v2).


**Args**

* **observation_space**  : Observation space of the environment.
* **action_space**  : Action shape of the environment.
* **action_type**  : Continuous or discrete action. "cont" or "dis".
* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **feature_dim**  : Number of features extracted.
* **lr**  : The learning rate.
* **eps**  : Term added to the denominator to improve numerical stability.
* **hidden_dim**  : The size of the hidden layers.
* **critic_target_tau**  : The critic Q-function soft-update rate.
* **update_every_steps**  : The agent update frequency.
* **num_init_steps**  : The exploration steps.
* **stddev_schedule**  : The exploration std schedule.
* **stddev_clip**  : The exploration std clip range.



**Returns**

DrQv2 learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L178)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L194)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L206)
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

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L233)
```python
.update(
   replay_buffer: DataLoader, step: int = 0
)
```

---
Update the learner.


**Args**

* **replay_buffer**  : Hsuanwu replay buffer.
* **step**  : Global training step.


**Returns**

Training metrics such as actor loss, critic_loss, etc.

### .update_critic
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L292)
```python
.update_critic(
   obs: Tensor, action: Tensor, reward: Tensor, discount: Tensor, next_obs: Tensor,
   step: int
)
```

---
Update the critic network.


**Args**

* **obs**  : Observations.
* **action**  : Actions.
* **reward**  : Rewards.
* **discount**  : discounts.
* **next_obs**  : Next observations.
* **step**  : Global training step.


**Returns**

Critic loss metrics.

### .update_actor
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/drqv2.py/#L339)
```python
.update_actor(
   obs: Tensor, step: int
)
```

---
Update the actor network.


**Args**

* **obs**  : Observations.
* **step**  : Global training step.


**Returns**

Actor loss metrics.
