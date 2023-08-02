#


## BaseStorage
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L76)
```python 
BaseStorage(
   observation_space: gym.Space, action_space: gym.Space, device: str = 'cpu'
)
```


---
Base class of storage module.


**Args**

* **observation_space** (gym.Space) : The observation space of environment.
* **action_space** (gym.Space) : The action space of environment.
* **device** (str) : Device (cpu, cuda, ...) on which the code should be run.


**Returns**

Instance of the base storage.


**Methods:**


### .to_torch
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L102)
```python
.to_torch(
   x: np.ndarray
)
```

---
Convert numpy array to torch tensor.


**Args**

* **x** (np.ndarray) : Numpy array.


**Returns**

Torch tensor.

### .add
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L114)
```python
.add(
   *args
)
```

---
Add samples to the storage.

### .sample
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L118)
```python
.sample(
   *args
)
```

---
Sample from the storage.

### .update
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/base_storage.py/#L122)
```python
.update(
   *args
)
```

---
Update the storage if necessary.
