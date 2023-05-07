#


## SAC
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L67)
```python 
SAC(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str, feature_dim: int, lr: float, eps: float,
   hidden_dim: int, critic_target_tau: float, update_every_steps: int,
   log_std_range: Tuple[float], betas: Tuple[float], temperature: float,
   fixed_temperature: bool, discount: float
)
```


---
Soft Actor-Critic (SAC) agent.
When 'augmentation' module is invoked, this learner will transform into Data Regularized Q (DrQ) agent.
Based on: https://github.com/denisyarats/pytorch_sac


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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L146)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L161)
```python
.integrate(
   **kwargs
)
```

---
Integrate agent and other modules (encoder, reward, ...) together

### .alpha
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L174)
```python
.alpha()
```

---
Get the temperature coefficient.

### .act
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L178)
```python
.act(
   obs: th.Tensor, training: bool = True, step: int = 0
)
```

---
Sample actions based on observations.


**Args**

* **obs** (Tensor) : Observations.
* **training** (bool) : training mode, True or False.
* **step** (int) : Global training step.


**Returns**

Sampled actions.

### .update
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L199)
```python
.update(
   replay_storage, step: int = 0
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L279)
```python
.update_critic(
   obs: th.Tensor, action: th.Tensor, reward: th.Tensor, terminated: th.Tensor,
   next_obs: th.Tensor, weights: th.Tensor, aug_obs: th.Tensor,
   aug_next_obs: th.Tensor, step: int
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
* **weights** (Tensor) : Batch sample weights.
* **aug_obs** (Tensor) : Augmented observations.
* **aug_next_obs** (Tensor) : Augmented next observations.
* **step** (int) : Global training step.


**Returns**

Critic loss metrics.

### .update_actor_and_alpha
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L353)
```python
.update_actor_and_alpha(
   obs: th.Tensor, weights: th.Tensor, step: int
)
```

---
Update the actor network and temperature.


**Args**

* **obs** (Tensor) : Observations.
* **weights** (Tensor) : Batch sample weights.
* **step** (int) : Global training step.


**Returns**

Actor loss metrics.

### .save
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L389)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/agent/sac.py/#L407)
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
