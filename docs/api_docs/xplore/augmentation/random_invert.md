#


## RandomInvert
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_invert.py/#L7)
```python 
RandomInvert(
   p: float = 0.5
)
```


---
RandomInvert method based on “RandomInvert: Inverts the colors of the given image randomly with a given probability”.

**Args**

* **p** (float) : probability of the image being color inverted. Default value is 0.5


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_invert.py/#L23)
```python
.forward(
   x: th.Tensor
)
```

