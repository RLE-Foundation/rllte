#


## RandomAdjustSharpness
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_adjustsharpness.py/#L7)
```python 
RandomAdjustSharpness(
   sharpness_factor: float = 50.0, p: float = 5.0
)
```


---
RandomAdjustSharpness method based on “RandomAdjustSharpness: Adjust the
sharpness of the image randomly with a given probability”.

**Args**

* **sharpness_factor** (float) : How much to adjust the sharpness. Can be any non-negative number. Default is 2.
* **p** (float) : probability of the image being sharpened. Default value is 0.5


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_adjustsharpness.py/#L28)
```python
.forward(
   x: th.Tensor
)
```

