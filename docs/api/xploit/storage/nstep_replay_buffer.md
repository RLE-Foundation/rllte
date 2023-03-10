#


## NStepReplayBuffer
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L11)
```python 
NStepReplayBuffer(
   buffer_size: int, batch_size: int, num_workers: int, pin_memory: bool,
   n_step: int = 2, discount: float = 0.99, fetch_every: int = 1000,
   save_snapshot: bool = False
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


### .add
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_buffer.py/#L62)
```python
.add(
   observation: Any, action: Any, reward: float, done: float, info: Any
)
```

