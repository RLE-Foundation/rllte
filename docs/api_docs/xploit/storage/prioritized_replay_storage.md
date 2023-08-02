#


## PrioritizedReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L36)
```python 
PrioritizedReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, num_envs: int = 1, batch_size: int = 1024,
   alpha: float = 0.6, beta: float = 0.4
)
```


---
Prioritized replay storage with proportional prioritization for off-policy algorithms.
Since the storage updates the priorities of the samples based on the TD error, users 
should include the `indices` and `weights` in the returned information of the `.update`
method of the agent. An example is:
    return {"indices": indices, "weights": weights, ..., "Actor Loss": actor_loss, ...}


**Args**

* **observation_space** (gym.Space) : Observation space.
* **action_space** (gym.Space) : Action space.
* **device** (str) : Device to store the data.
* **storage_size** (int) : Storage size.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size.
* **alpha** (float) : Prioritization value.
* **beta** (float) : Importance sampling value.


**Returns**

Prioritized replay storage.


**Methods:**


### .annealing_beta
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L85)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L96)
```python
.add(
   observations: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, info: Dict[str, Any],
   next_observations: th.Tensor
)
```

---
Add sampled transitions into storage.


**Args**

* **observations** (th.Tensor) : Observations.
* **actions** (th.Tensor) : Actions.
* **rewards** (th.Tensor) : Rewards.
* **terminateds** (th.Tensor) : Termination flag.
* **truncateds** (th.Tensor) : Truncation flag.
* **info** (Dict[str, Any]) : Additional information.
* **next_observations** (th.Tensor) : Next observations.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L134)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L177)
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
