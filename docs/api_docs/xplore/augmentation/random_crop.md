#


## RandomCrop
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_crop.py/#L7)
```python 
RandomCrop(
   pad: int = 4, out: int = 84
)
```


---
Random crop operation for processing image-based observations.


**Args**

* **pad** (int) : Padding size.
* **out** (int) : Desired output size.


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_crop.py/#L23)
```python
.forward(
   x: th.Tensor
)
```

