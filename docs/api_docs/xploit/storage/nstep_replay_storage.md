#


## NStepReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L246)
```python 
NStepReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, num_envs: int = 1, batch_size: int = 256,
   num_workers: int = 4, pin_memory: bool = True, n_step: int = 3, discount: float = 0.99,
   fetch_every: int = 1000, save_snapshot: bool = False
)
```


---
N-step replay storage.
Implemented based on: https://github.com/facebookresearch/drqv2/blob/main/replay_buffer.py


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to convert replay data.
* **storage_size** (int) : Max number of element in the storage.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size of samples.
* **num_workers** (int) : Subprocesses to use for data loading.
* **pin_memory** (bool) : Pin memory or not.
* **nstep** (int) : The number of transitions to consider when computing n-step returns
* **discount** (float) : The discount factor for future rewards.
* **fetch_every** (int) : Loading interval.
* **save_snapshot** (bool) : Save loaded file or not.


**Returns**

N-step replay storage.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L302)
```python
.reset()
```

---
Reset the storage.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L313)
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

### .replay_iter
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L349)
```python
.replay_iter()
```

---
Create iterable dataloader.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L355)
```python
.sample()
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L368)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
