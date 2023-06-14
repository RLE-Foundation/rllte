#


## NStepReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L246)
```python 
NStepReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 256, num_workers: int = 4,
   pin_memory: bool = True, n_step: int = 3, discount: float = 0.99,
   fetch_every: int = 1000, save_snapshot: bool = False
)
```


---
N-step replay storage.
Implemented based on: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to store replay data.
* **storage_size** (int) : Max number of element in the storage.
* **batch_size** (int) : Batch size.
* **num_workers** (int) : Subprocesses to use for data loading.
* **pin_memory** (bool) : Pin memory or not.
* **nstep** (int) : The number of transitions to consider when computing n-step returns
* **discount** (float) : The discount factor for future rewards.
* **fetch_every** (int) : Loading interval.
* **save_snapshot** (bool) : Save loaded file or not.


**Returns**

N-step replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L304)
```python
.add(
   obs: th.Tensor, action: th.Tensor, reward: th.Tensor, terminated: th.Tensor,
   truncated: th.Tensor, info: th.Tensor, next_obs: th.Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **obs** (th.Tensor) : Observation.
* **action** (th.Tensor) : Action.
* **reward** (th.Tensor) : Reward.
* **terminated** (th.Tensor) : Termination flag.
* **truncated** (th.Tensor) : Truncation flag.
* **info** (th.Tensor) : Additional information.
* **next_obs** (th.Tensor) : Next observation.


**Returns**

None.

### .replay_iter
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L340)
```python
.replay_iter()
```

---
Create iterable dataloader.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L346)
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

Sampled data.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L357)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
