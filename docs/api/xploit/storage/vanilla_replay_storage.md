#


## VanillaReplayStorage
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_replay_storage.py\#L7)
```python 
VanillaReplayStorage(
   device: Device, obs_shape: Tuple, action_shape: Tuple, action_type: str,
   storage_size: int = 1000000.0, batch_size: int = 1024
)
```


---
Vanilla replay storage for off-policy algorithms.


**Args**

* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **obs_shape** (Tuple) : The data shape of observations.
* **action_shape** (Tuple) : The data shape of actions.
* **action_type** (str) : The type of actions, 'Discrete' or 'Box'.
* **storage_size** (int) : Max number of element in the buffer.
* **batch_size** (int) : Batch size of samples.


**Returns**

Vanilla replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_replay_storage.py\#L56)
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
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/xploit/storage/vanilla_replay_storage.py\#L87)
```python
.sample()
```

---
Sample transitions from the storage.


**Args**

None.


**Returns**

Batched samples.
