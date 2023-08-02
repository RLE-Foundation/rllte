#


## DQN
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/dqn.py/#L39)
```python 
DQN(
   env: gym.Env, eval_env: Optional[gym.Env] = None, tag: str = 'default', seed: int = 1,
   device: str = 'cpu', pretraining: bool = False, num_init_steps: int = 2000,
   eval_every_steps: int = 5000, feature_dim: int = 50, batch_size: int = 32,
   lr: float = 0.001, eps: float = 1e-08, hidden_dim: int = 1024, tau: float = 1.0,
   update_every_steps: int = 4, target_update_freq: int = 1000, discount: float = 0.99,
   init_fn: str = 'orthogonal'
)
```


---
Deep Q-Network (DQN) agent.


**Args**

* **env** (gym.Env) : A Gym-like environment for training.
* **eval_env** (gym.Env) : A Gym-like environment for evaluation.
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
* **tau**  : The Q-function soft-update rate.
* **update_every_steps** (int) : The update frequency of the policy.
* **target_update_freq** (int) : The frequency of target Q-network update.
* **discount** (float) : Discount factor.
* **init_fn** (str) : Parameters initialization method.



**Returns**

DQN agent instance.


**Methods:**


### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/agent/legacy/dqn.py/#L138)
```python
.update()
```

---
Update the agent and return training metrics such as actor loss, critic_loss, etc.
