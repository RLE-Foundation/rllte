#


## DictReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L36)
```python 
DictReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 1024, num_envs: int = 1
)
```


---
Dict replay storage for off-policy algorithms and dictionary observations.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to convert the data.
* **storage_size** (int) : The capacity of the storage.
* **batch_size** (int) : Batch size of samples.
* **num_envs** (int) : The number of parallel environments.


**Returns**

Dict replay storage.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L67)
```python
.reset()
```

---
Reset the storage.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L79)
```python
.add(
   observations: Dict[str, th.Tensor], actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, infos: Dict[str, Any],
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
* **infos** (Dict[str, Any]) : Additional information.
* **next_observations** (Dict[str, th.Tensor]) : Next observations.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L122)
```python
.sample()
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/dict_replay_storage.py/#L156)
```python
.update(
   *args, **kwargs
)
```

---
Update the storage if necessary.
