#


## SACLearner
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/sac.py\#L62)
```python 
SACLearner(
   observation_space: Dict, action_space: Dict, device: Device, feature_dim: int,
   lr: float, eps: float, hidden_dim: int, critic_target_tau: float,
   update_every_steps: int, log_std_range: Tuple[float], betas: Tuple[float],
   temperature: float, fixed_temperature: bool, discount: float
)
```


---
Soft Actor-Critic (SAC) Learner.
When 'augmentation' module is invoked, this learner will transform into Data Regularized Q (DrQ) Learner.


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
* **critic_target_tau** (float) : The critic Q-function soft-update rate.
* **update_every_steps** (int) : The agent update frequency.
* **log_std_range** (Tuple[float]) : Range of std for sampling actions.
* **betas** (Tuple[float]) : coefficients used for computing running averages of gradient and its square.
* **temperature** (float) : Initial temperature coefficient.
* **fixed_temperature** (bool) : Fixed temperature or not.
* **discount** (float) : Discount factor.



**Returns**

Soft Actor-Critic learner instance.


**Methods:**


### .train
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/sac.py\#L148)
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

### .alpha
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/sac.py\#L164)
```python
.alpha()
```

---
Get the temperature coefficient.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/sac.py\#L168)
```python
.update(
   replay_storage: Storage, step: int = 0
)
```

---
Update the learner.


**Args**

* **replay_storage** (Storage) : Hsuanwu replay storage.
* **step** (int) : Global training step.


**Returns**

Training metrics such as actor loss, critic_loss, etc.

### .update_critic
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/sac.py\#L234)
```python
.update_critic(
   obs: Tensor, action: Tensor, reward: Tensor, terminated: Tensor, next_obs: Tensor,
   aug_obs: Tensor, aug_next_obs: Tensor, step: int
)
```

---
Update the critic network.


**Args**

* **obs** (Tensor) : Observations.
* **action** (Tensor) : Actions.
* **reward** (Tensor) : Rewards.
* **terminated** (Tensor) : Terminateds.
* **next_obs** (Tensor) : Next observations.
* **aug_obs** (Tensor) : Augmented observations.
* **aug_next_obs** (Tensor) : Augmented next observations.
* **step** (int) : Global training step.


**Returns**

Critic loss metrics.

### .update_actor_and_alpha
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/learner/sac.py\#L302)
```python
.update_actor_and_alpha(
   obs: Tensor, step: int
)
```

---
Update the actor network and temperature.


**Args**

* **obs** (Tensor) : Observations.
* **step** (int) : Global training step.


**Returns**

Actor loss metrics.
