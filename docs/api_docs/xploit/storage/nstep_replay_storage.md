#


## NStepReplayStorage
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_storage.py/#L191)
```python 
NStepReplayStorage(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str = 'cpu', storage_size: int = 500000, batch_size: int = 256,
   num_workers: int = 4, pin_memory: bool = True, n_step: int = 3, discount: float = 0.99,
   fetch_every: int = 1000, save_snapshot: bool = False
)
```


---
Replay storage for off-policy algorithms (N-step returns supported).


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_storage.py/#L252)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_storage.py/#L284)
```python
.replay_iter()
```

---
Create iterable dataloader.

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_storage.py/#L290)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/nstep_replay_storage.py/#L301)
```python
.update(
   *args
)
```

---
Update the storage
