#


## MiniGrid
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/minigrid.py/#L38)
```python 

```


---
Scores and learning cures of various RL algorithms on the MiniGrid benchmark.
Environment link: https://github.com/Farama-Foundation/Minigrid
Number of environments: 16
Number of training steps: 1,000,000
Number of seeds: 10
Added algorithms: [A2C]


**Methods:**


### .load_scores
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/minigrid.py/#L52)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/minigrid.py/#L75)
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


**Returns**

* **train**  : np.ndarray(shape=(N_SEEDS, N_POINTS))
* **eval**  :  np.ndarray(shape=(N_SEEDS, N_POINTS))
Learning curves data with structure:
curves

### .load_models
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/minigrid.py/#L105)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/minigrid.py/#L132)
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
