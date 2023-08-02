#


## HerReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/her_replay_storage.py/#L37)
```python 
HerReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, num_envs: int = 1, batch_size: int = 1024,
   goal_selection_strategy: str = 'future', num_goals: int = 4,
   reward_fn: Callable = None, copy_info_dict: bool = False
)
```


---
Hindsight experience replay (HER) storage for off-policy algorithms.
Based on: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to store the data.
* **storage_size** (int) : Storage size.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size.
* **goal_selection_strategy** (str) : A goal selection strategy of ["future", "final", "episode"].
* **num_goals** (int) : The number of goals to sample.
* **reward_fn** (Callable) : Function to compute new rewards based on state and goal, whose definition is
    same as https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/bit_flipping_env.py#L190
copy_info_dict (bool) whether to copy the info dictionary and pass it to compute_reward() method. 


**Returns**

Dict replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/her_replay_storage.py/#L97)
```python
.add(
   observations: Dict[str, th.Tensor], actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, info: Dict[str, Any],
   next_observations: Dict[str, th.Tensor]
)
```

---
Add sampled transitions into storage.


**Args**

* **observations** (Dict[str, th.Tensor]) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **terminateds** (th.Tensor) : Termination flag.
* **truncateds** (th.Tensor) : Truncation flag.
* **info** (Dict[str, Any]) : Additional information.
* **next_observations** (Dict[str, th.Tensor]) : Next observations.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/her_replay_storage.py/#L163)
```python
.sample(
   step: int
)
```

---
Sample from the storage.


**Args**

* **step** (int) : Global training step.


**Returns**

Batched samples.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/her_replay_storage.py/#L342)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
