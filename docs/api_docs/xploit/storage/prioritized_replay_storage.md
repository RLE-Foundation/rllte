#


## PrioritizedReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L36)
```python 
PrioritizedReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 1024, alpha: float = 0.6,
   beta: float = 0.4
)
```


---
Prioritized replay storage with proportional prioritization for off-policy algorithms.


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to store the data.
* **storage_size** (int) : Storage size.
* **batch_size** (int) : Batch size.
* **alpha** (float) : Prioritization value.
* **beta** (float) : Importance sampling value.


**Returns**

Prioritized replay storage.


**Methods:**


### .annealing_beta
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L75)
```python
.annealing_beta(
   step: int
)
```

---
Linearly increases beta from the initial value to 1 over global training steps.


**Args**

* **step** (int) : The global training step.


**Returns**

Beta value.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L86)
```python
.add(
   obs: th.Tensor, action: th.Tensor, reward: th.Tensor, terminated: th.Tensor,
   truncated: th.Tensor, info: th.Tensor, next_obs: th.Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **obs** (th.Tensor) : Observation.
* **action** (th.Tensor) : Action.
* **reward** (th.Tensor) : Reward.
* **terminated** (th.Tensor) : Termination flag.
* **truncated** (th.Tensor) : Truncation flag.
* **info** (th.Tensor) : Additional information.
* **next_obs** (th.Tensor) : Next observation.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L118)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L159)
```python
.update(
   metrics: Dict
)
```

---
Update the priorities.


**Args**

* **metrics** (Dict) : Training metrics from agent to udpate the priorities:
    indices (np.ndarray): The indices of current batch data.
    priorities (np.ndarray): The priorities of current batch data.


**Returns**

None.
