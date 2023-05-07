#


## VanillaReplayStorage
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L11)
```python 
VanillaReplayStorage(
   observation_space: Union[gym.Space, DictConfig], action_space: Union[gym.Space,
   DictConfig], device: str = 'cpu', storage_size: int = 1000000, batch_size: int = 1024
)
```


---
Vanilla replay storage for off-policy algorithms.


**Args**

* **observation_space** (Space or DictConfig) : The observation space of environment. When invoked by Hydra,
    'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
* **action_space** (Space or DictConfig) : The action space of environment. When invoked by Hydra,
    'action_space' is a 'DictConfig' like
    {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
    {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **storage_size** (int) : Max number of element in the buffer.
* **batch_size** (int) : Batch size of samples.


**Returns**

Vanilla replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L60)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L91)
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
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L115)
```python
.update(
   *args
)
```

---
Update the storage
