#


## NStepReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L198)
```python 
NStepReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 256, num_workers: int = 4,
   pin_memory: bool = True, n_step: int = 3, discount: float = 0.99,
   fetch_every: int = 1000, save_snapshot: bool = False
)
```


---
Replay storage for off-policy algorithms (N-step returns supported).


**Args**

* **observation_space** (Space) : The observation space of environment.
* **action_space** (Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **storage_size** (int) : Max number of element in the storage.
* **batch_size** (int) : Number of samples per batch to load.
* **num_workers** (int) : Subprocesses to use for data loading.
* **pin_memory** (bool) : Copy Tensors into device/CUDA pinned memory before returning them.
* **discount** (float) : The discount factor for future rewards.
* **fetch_every** (int) : Loading interval.
* **save_snapshot** (bool) : Save loaded file or not.
n_step (int) The number of transitions to consider when computing n-step returns


**Returns**

N-step replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L255)
```python
.add(
   obs: Any, action: Any, reward: Any, terminated: Any, info: Any, next_obs: Any
)
```

---
Add sampled transitions into storage.


**Args**

* **obs** (Any) : Observations.
* **action** (Any) : Actions.
* **reward** (Any) : Rewards.
* **terminated** (Any) : Terminateds.
* **info** (Any) : Infos.
* **next_obs** (Any) : Next observations.


**Returns**

None.

### .replay_iter
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L287)
```python
.replay_iter()
```

---
Create iterable dataloader.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L293)
```python
.sample(
   step: int
)
```

---
Generate samples.


**Args**

* **step** (int) : Global training step.


**Returns**

Batched samples.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/nstep_replay_storage.py/#L304)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
