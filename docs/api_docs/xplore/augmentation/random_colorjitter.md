#


## RandomColorJitter
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_colorjitter.py/#L7)
```python 
RandomColorJitter(
   brightness: float = 0.4, contrast: float = 0.4, saturation: float = 0.4,
   hue: float = 0.5
)
```


---
Random ColorJitter operation for image augmentation.


**Args**

* **brightness** (float) : How much to jitter brightness. Should be non negative numbers.
* **contrast** (float) : How much to jitter contrast. Should be non negative numbers.
* **saturation** (float) : How much to jitter saturation. Should be non negative numbers.
* **hue** (float) : How much to jitter hue. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_colorjitter.py/#L30)
```python
.forward(
   x: th.Tensor
)
```

