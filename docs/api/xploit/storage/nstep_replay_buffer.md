#


## NStepReplayBuffer
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L63)
```python 
NStepReplayBuffer(
   buffer_size: int = 500000, batch_size: int = 256, num_workers: int = 4,
   pin_memory: bool = True, n_step: int = 3, discount: float = 0.99,
   fetch_every: int = 1000, save_snapshot: bool = False
)
```


---
Replay buffer for off-policy algorithms (N-step returns supported).


**Args**

* **buffer_size**  : Max number of element in the buffer.
* **batch_size**  : Number of samples per batch to load.
* **num_workers**  : Subprocesses to use for data loading.
* **pin_memory**  : Copy Tensors into device/CUDA pinned memory before returning them.
* **n_step**  : The number of transitions to consider when computing n-step returns
* **discount**  : The discount factor for future rewards.
* **fetch_every**  : Loading interval.
* **save_snapshot**  : Save loaded file or not.


**Returns**

N-step replay buffer.


**Methods:**


### .get_batch_size
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L110)
```python
.get_batch_size()
```


### .get_num_workers
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L115)
```python
.get_num_workers()
```


### .get_pin_memory
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L120)
```python
.get_pin_memory()
```


### .add
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L129)
```python
.add(
   obs: Any, action: Any, reward: float, done: float, info: Dict, next_obs: Any
)
```

