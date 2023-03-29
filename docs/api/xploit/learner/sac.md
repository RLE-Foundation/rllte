#


## SACLearner
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L117)
```python 
SACLearner(
   observation_space: Space, action_space: Space, action_type: str,
   device: torch.device = 'cuda', feature_dim: int = 5, lr: float = 0.0001,
   eps: float = 8e-05, hidden_dim: int = 1024, critic_target_tau: float = 0.005,
   num_init_steps: int = 5000, update_every_steps: int = 2,
   log_std_range: Tuple[float] = (-5.0, 2), betas: Tuple[float] = (0.9, 0.999),
   temperature: float = 0.1, fixed_temperature: bool = False, discount: float = 0.99
)
```


---
Soft Actor-Critic (SAC) Learner


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
* **log_std_range**  : Range of std for sampling actions.
* **betas**  : coefficients used for computing running averages of gradient and its square.
* **temperature**  : Initial temperature coefficient.
* **fixed_temperature**  : Fixed temperature or not.
* **discount**  : Discount factor.



**Returns**

Soft Actor-Critic learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L198)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L214)
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

### ._alpha
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L227)
```python
._alpha()
```

---
Get the temperature coefficient.


### .act
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L233)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L259)
```python
.update(
   replay_buffer: Generator, step: int = 0
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L310)
```python
.update_critic(
   obs: Tensor, action: Tensor, reward: Tensor, done: Tensor, next_obs: Tensor,
   step: int
)
```

---
Update the critic network.


**Args**

* **obs**  : Observations.
* **action**  : Actions.
* **reward**  : Rewards.
* **done**  : Dones.
* **next_obs**  : Next observations.
* **step**  : Global training step.


**Returns**

Critic loss metrics.

### .update_actor_and_alpha
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/learner/sac.py/#L354)
```python
.update_actor_and_alpha(
   obs: Tensor, step: int
)
```

---
Update the actor network and temperature.


**Args**

* **obs**  : Observations.
* **step**  : Global training step.


**Returns**

Actor loss metrics.
