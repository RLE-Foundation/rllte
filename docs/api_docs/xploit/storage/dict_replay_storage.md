#


## DictReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L35)
```python 
DictReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, num_envs: int = 1, batch_size: int = 1024
)
```


---
Dict replay storage for off-policy algorithms and dictionary observations.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to store the data.
* **storage_size** (int) : Storage size.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size.


**Returns**

Dict replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L71)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L115)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L153)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
