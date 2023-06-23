#


## VanillaReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L35)
```python 
VanillaReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 1024
)
```


---
Vanilla replay storage for off-policy algorithms.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to store the data.
* **storage_size** (int) : Storage size.
* **batch_size** (int) : Batch size.


**Returns**

Vanilla replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L80)
```python
.add(
   obs: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
   truncated: np.ndarray, info: Dict[str, Any], next_obs: np.ndarray
)
```

---
Add sampled transitions into storage.


**Args**

* **obs** (np.ndarray) : Observation.
* **action** (np.ndarray) : Action.
* **reward** (np.ndarray) : Reward.
* **terminated** (np.ndarray) : Termination flag.
* **truncated** (np.ndarray) : Truncation flag.
* **info** (Dict[str, Any]) : Additional information.
* **next_obs** (np.ndarray) : Next observation.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L114)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/vanilla_replay_storage.py/#L139)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
