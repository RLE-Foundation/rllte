#


## PrioritizedReplayStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L38)
```python 
PrioritizedReplayStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu',
   storage_size: int = 1000000, batch_size: int = 1024, num_envs: int = 1,
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
* **device** (str) : Device to convert the data.
* **storage_size** (int) : The capacity of the storage.
* **num_envs** (int) : The number of parallel environments.
* **batch_size** (int) : Batch size of samples.
* **alpha** (float) : Prioritization value.
* **beta** (float) : Importance sampling value.


**Returns**

Prioritized replay storage.


**Methods:**


### .reset
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L78)
```python
.reset()
```

---
Reset the storage.

### .annealing_beta
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L90)
```python
.annealing_beta()
```

---
Linearly increases beta from the initial value to 1 over global training steps.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L94)
```python
.add(
   observations: th.Tensor, actions: th.Tensor, rewards: th.Tensor,
   terminateds: th.Tensor, truncateds: th.Tensor, infos: Dict[str, Any],
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
* **infos** (Dict[str, Any]) : Additional information.
* **next_observations** (th.Tensor) : Next observations.


**Returns**

None.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L135)
```python
.sample()
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xploit/storage/prioritized_replay_storage.py/#L173)
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
