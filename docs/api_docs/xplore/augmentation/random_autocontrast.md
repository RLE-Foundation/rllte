#


## RandomAutocontrast
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_autocontrast.py/#L7)
```python 
RandomAutocontrast(
   p: float = 0.5
)
```


---
RandomAutocontrast method based on “RandomAutocontrast:
Autocontrast the pixels of the given image randomly with a given probability”.

**Args**

* **p** (float) : probability of the image being autocontrasted. Default value is 0.5


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_autocontrast.py/#L25)
```python
.forward(
   x: th.Tensor
)
```

