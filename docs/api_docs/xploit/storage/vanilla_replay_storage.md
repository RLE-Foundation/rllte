#


## VanillaReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L36)
```python 
VanillaReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 1024, num_envs: int = 1
)
```


---
Vanilla replay storage for off-policy algorithms.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to convert the data.
* **storage_size** (int) : The capacity of the storage.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size of samples.


**Returns**

Vanilla replay storage.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L65)
```python
.reset()
```

---
Reset the storage.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L81)
```python
.add(
   observations: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, infos: Dict[str, Any],
   next_observations: th.Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **observations** (th.Tensor) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **terminateds** (th.Tensor) : Termination flag.
* **truncateds** (th.Tensor) : Truncation flag.
* **infos** (Dict[str, Any]) : Additional information.
* **next_observations** (th.Tensor) : Next observations.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L116)
```python
.sample()
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L143)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
