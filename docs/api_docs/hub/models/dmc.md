#


## DMControl
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/dmc.py/#L32)
```python 

```


---
Scores and learning cures of various RL algorithms on the full
DeepMind Control Suite benchmark.
---
Environment link: https://github.com/google-deepmind/dm_control
Number of environments: 24
Number of training steps: 1,000,000
Number of seeds: 10
Added algorithms: [SAC, DrQ-v2]


**Methods:**


### .load_scores
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/dmc.py/#L45)
```python
.load_scores()
```

---
Returns final performance.

### .load_curves
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/hub/datasets/dmc.py/#L56)
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
