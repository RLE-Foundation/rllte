#


## DMControl
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/dmc.py/#L41)
```python 

```


---
Scores and learning cures of various RL algorithms on the full
DeepMind Control Suite benchmark.
---
Environment link: https://github.com/google-deepmind/dm_control
Number of environments: 27
Number of training steps: 10,000,000 for humanoid, 2,000,000 for others
Number of seeds: 10
Added algorithms: [SAC, DrQ-v2]


**Methods:**


### .get_obs_type
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/dmc.py/#L65)
```python
.get_obs_type(
   agent: str
)
```

---
Returns the observation type of the agent.


**Args**

* **agent** (str) : Agent name.


**Returns**

Observation type.

### .load_scores
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/dmc.py/#L77)
```python
.load_scores(
   env_id: str, agent: str
)
```

---
Returns final performance.


**Args**

* **env_id** (str) : Environment ID.
* **agent_id** (str) : Agent name.


**Returns**

Test scores data array with shape (N_SEEDS, N_POINTS).

### .load_curves
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/dmc.py/#L101)
```python
.load_curves(
   env_id: str, agent: str
)
```

---
Returns learning curves using a `Dict` of NumPy arrays.


**Args**

* **env_id** (str) : Environment ID.
* **agent_id** (str) : Agent name.
* **obs_type** (str) : A type from ['state', 'pixel'].


**Returns**

* **train**  : np.ndarray(shape=(N_SEEDS, N_POINTS))
* **eval**  :  np.ndarray(shape=(N_SEEDS, N_POINTS))
Learning curves data with structure:
curves

### .load_models
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/dmc.py/#L132)
```python
.load_models(
   env_id: str, agent: str, seed: int, device: str = 'cpu'
)
```

---
Load the model from the hub.


**Args**

* **env_id** (str) : Environment ID.
* **agent** (str) : Agent name.
* **seed** (int) : The seed to load.
* **device** (str) : The device to load the model on.


**Returns**

The loaded model.

### .load_apis
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/dmc.py/#L160)
```python
.load_apis(
   env_id: str, agent: str, seed: int, device: str = 'cpu'
)
```

---
Load the a training API.


**Args**

* **env_id** (str) : Environment ID.
* **agent** (str) : Agent name.
* **seed** (int) : The seed to load.
* **device** (str) : The device to load the model on.


**Returns**

The loaded API.
