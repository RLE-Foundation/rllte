#


## DictRolloutStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_rollout_storage.py/#L35)
```python 
DictRolloutStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   num_steps: int = 256, num_envs: int = 8, batch_size: int = 64, discount: float = 0.999,
   gae_lambda: float = 0.95
)
```


---
Dict Rollout storage for on-policy algorithms and dictionary observations.


**Args**

* **observation_space** (gym.Space) : The observation space of environment.
* **action_space** (gym.Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **num_steps** (int) : The sample length of per rollout.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size of samples.
* **discount** (float) : discount factor.
* **gae_lambda** (float) : Weighting coefficient for generalized advantage estimation (GAE).


**Returns**

Dict rollout storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_rollout_storage.py/#L78)
```python
.add(
   observations: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, info: Dict,
   next_observations: th.Tensor, log_probs: th.Tensor, values: th.Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **observations** (th.Tensor) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **terminateds** (th.Tensor) : Termination signals.
* **truncateds** (th.Tensor) : Truncation signals.
* **info** (Dict) : Extra information.
* **next_observations** (th.Tensor) : Next observations.
* **log_probs** (th.Tensor) : Log of the probability evaluated at `actions`.
* **values** (th.Tensor) : Estimated values.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_rollout_storage.py/#L126)
```python
.sample()
```

---
Sample data from storage.
