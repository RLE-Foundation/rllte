#


## RandomEqualize
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_equalize.py/#L7)
```python 
RandomEqualize(
   p: float = 0.5
)
```


---
RandomEqualize method based on “RandomEqualize: Equalize the
histogram of the given image randomly with a given probability”.

**Args**

* **p** (float) : probability of the image being equalized. Default value is 0.5


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_equalize.py/#L25)
```python
.forward(
   x: th.Tensor
)
```

