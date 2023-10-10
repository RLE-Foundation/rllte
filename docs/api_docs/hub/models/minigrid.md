#


## MiniGrid
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/minigrid.py/#L32)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/minigrid.py/#L44)
```python
.load_scores()
```

---
Returns final performance.

### .load_curves
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/minigrid.py/#L55)
```python
.load_curves()
```

---
Returns learning curves using a `Dict` of NumPy arrays:
curves = {
    "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
},
    "eval": {"bigfish": np.ndarray(shape=(Number of seeds, Number of points)), ...},
},
...
---
}
