#


## NStepReplayStorage
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/nstep_replay_storage.py\#L69)
```python 
NStepReplayStorage(
   storage_size: int = 500000, batch_size: int = 256, num_workers: int = 4,
   pin_memory: bool = True, n_step: int = 3, discount: float = 0.99,
   fetch_every: int = 1000, save_snapshot: bool = False
)
```


---
Replay storage for off-policy algorithms (N-step returns supported).


**Args**

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


### .get_batch_size
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/nstep_replay_storage.py\#L117)
```python
.get_batch_size()
```


### .get_num_workers
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/nstep_replay_storage.py\#L121)
```python
.get_num_workers()
```


### .get_pin_memory
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/nstep_replay_storage.py\#L125)
```python
.get_pin_memory()
```


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/nstep_replay_storage.py\#L132)
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

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/nstep_replay_storage.py\#L207)
```python
.sample()
```

---
Generate samples.


**Args**

None.


**Returns**

Batched samples.
