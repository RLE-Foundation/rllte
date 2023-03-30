#


## VanillaReplayStorage
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L7)
```python 
VanillaReplayStorage(
   device: torch.device, obs_shape: Tuple, action_shape: Tuple, action_type: str,
   buffer_size: int = 1000000.0, batch_size: int = 1024
)
```


---
Vanilla replay storage for off-policy algorithms.


**Args**

* **device**  : Device (cpu, cuda, ...) on which the code should be run.
* **obs_shape**  : The data shape of observations.
* **action_shape**  : The data shape of actions.
* **action_type**  : The type of actions, 'cont' or 'dis'.
* **buffer_size**  : Max number of element in the buffer.
* **batch_size**  : Batch size of samples.


**Returns**

Vanilla replay storage.


**Methods:**


### .add
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L56)
```python
.add(
   obs: Any, action: Any, reward: Any, done: Any, info: Any, next_obs: Any
)
```

---
Add sampled transitions into storage.


**Args**

* **obs**  : Observations.
* **actions**  : Actions.
* **rewards**  : Rewards.
* **dones**  : Dones.
* **info**  : Infos.
* **next_obs**  : Next observations.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xploit/storage/vanilla_replay_storage.py/#L81)
```python
.sample()
```

---
Sample transitions from the storage.
