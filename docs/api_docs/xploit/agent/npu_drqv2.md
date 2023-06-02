#


## NpuDrQv2
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L22)
```python 
NpuDrQv2(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_init_steps: int = 2000,
   eval_every_steps: int = 5000, feature_dim: int = 50, batch_size: int = 256,
   lr: float = 0.0001, eps: float = 1e-08, hidden_dim: int = 1024,
   critic_target_tau: float = 0.01, update_every_steps: int = 2,
   network_init_method: str = 'orthogonal'
)
```


---
Data Regularized-Q v2 (DrQ-v2) for `NPU` device.
When 'augmentation' module is deprecated, this agent will transform into
    Deep Deterministic Policy Gradient (DDPG) agent.
Based on: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py


**Args**

* **env** (Env) : A Gym-like environment for training.
* **eval_env** (Env) : A Gym-like environment for evaluation.
* **tag** (str) : An experiment tag.
* **seed** (int) : Random seed for reproduction.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **pretraining** (bool) : Turn on the pre-training mode.
* **num_init_steps** (int) : Number of initial exploration steps.
* **eval_every_steps** (int) : Evaluation interval.
* **feature_dim** (int) : Number of features extracted by the encoder.
* **batch_size** (int) : Number of samples per batch to load.
* **lr** (float) : The learning rate.
* **eps** (float) : Term added to the denominator to improve numerical stability.
* **hidden_dim** (int) : The size of the hidden layers.
* **critic_target_tau**  : The critic Q-function soft-update rate.
* **update_every_steps** (int) : The agent update frequency.
* **network_init_method** (str) : Network initialization method name.



**Returns**

DrQv2 agent instance.


**Methods:**


### .mode
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L129)
```python
.mode(
   training: bool = True
)
```

---
Set the training mode.


**Args**

* **training** (bool) : True (training) or False (testing).


**Returns**

None.

### .set
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L144)
```python
.set(
   encoder: Optional[Any] = None, storage: Optional[Any] = None,
   distribution: Optional[Any] = None, augmentation: Optional[Any] = None,
   reward: Optional[Any] = None
)
```

---
Set a module for the agent.


**Args**

* **encoder** (Optional[Any]) : An encoder of `rllte.xploit.encoder` or a custom encoder.
* **storage** (Optional[Any]) : A storage of `rllte.xploit.storage` or a custom storage.
* **distribution** (Optional[Any]) : A distribution of `rllte.xplore.distribution` or a custom distribution.
* **augmentation** (Optional[Any]) : An augmentation of `rllte.xplore.augmentation` or a custom augmentation.
* **reward** (Optional[Any]) : A reward of `rllte.xplore.reward` or a custom reward.


**Returns**

None.

### .freeze
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L174)
```python
.freeze()
```

---
Freeze the structure of the agent.

### .act
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L197)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L218)
```python
.update()
```

---
Update the agent and return training metrics such as actor loss, critic_loss, etc.


### .update_critic
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L264)
```python
.update_critic(
   obs: th.Tensor, action: th.Tensor, reward: th.Tensor, discount: th.Tensor,
   next_obs: th.Tensor
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


**Returns**

Critic loss metrics.

### .update_actor
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L311)
```python
.update_actor(
   obs: th.Tensor
)
```

---
Update the actor network.


**Args**

* **obs** (Tensor) : Observations.


**Returns**

Actor loss metrics.

### .save
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L336)
```python
.save()
```

---
Save models.

### .load
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/agent/npu_drqv2.py/#L352)
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
