#


## ReplayBufferStorage
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_buffer.py/#L11)
```python 
ReplayBufferStorage(
   replay_dir: Path
)
```


---
Storage collected experiences to local files.


**Args**

* **replay_dir**  : save directory.


**Returns**

Storage instance.


**Methods:**


### .num_episodes
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_buffer.py/#L38)
```python
.num_episodes()
```


### .num_transitions
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_buffer.py/#L42)
```python
.num_transitions()
```


### .add
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_buffer.py/#L45)
```python
.add(
   obs: Any, action: Any, reward: float, done: bool, info: Any, use_discount: bool
)
```


----


## VanillaReplayBuffer
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_buffer.py/#L63)
```python 
VanillaReplayBuffer(
   buffer_size: int, num_workers: int, discount: float = 0.99, fetch_every: int = 1000,
   save_snapshot: bool = False
)
```


---
Vanilla replay buffer for off-policy algorithms.


**Args**

* **buffer_size**  : Max number of element in the buffer.
* **num_workers**  : Subprocesses to use for data loading.
* **discount**  : The discount factor for future rewards.
* **fetch_every**  : Loading interval.
* **save_snapshot**  : Save loaded file or not.


**Returns**

Vanilla replay buffer.


**Methods:**


### .add
[source](https://github.com/BellmanProject/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_buffer.py/#L105)
```python
.add(
   observation: Any, action: Any, reward: float, done: float, info: Any
)
```

