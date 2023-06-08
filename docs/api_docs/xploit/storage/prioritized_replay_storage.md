#


## PrioritizedReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L11)
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

* **observation_space** (Space) : The observation space of environment.
* **action_space** (Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.
* **storage_size** (int) : Max number of element in the buffer.
* **batch_size** (int) : Batch size of samples.
* **alpha** (float) : The alpha coefficient.
* **beta** (float) : The beta coefficient.


**Returns**

Prioritized replay storage.


**Methods:**


### .annealing_beta
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L50)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L61)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L89)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L128)
```python
.update(
   metrics: Dict
)
```

---
Update the priorities.


**Args**

* **metrics** (Dict) : Training metrics from agent to udpate the priorities:
* **indices** (NdArray) : The indices of current batch data.
* **priorities** (NdArray) : The priorities of current batch data.


**Returns**

None.
