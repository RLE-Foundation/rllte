#


## RandomCutout
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_cutout.py/#L8)
```python 
RandomCutout(
   min_cut: int = 10, max_cut: int = 30
)
```


---
Random Cutout operation for image augmentation.

**Args**

* **min_cut**  : Min size of the cut shape.
* **max_cut**  : Max size of the cut shape.


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_cutout.py/#L25)
```python
.forward(
   x: Tensor
)
```

