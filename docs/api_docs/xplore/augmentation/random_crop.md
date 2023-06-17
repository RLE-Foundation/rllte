#


## RandomCrop
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/augmentation/random_crop.py/#L32)
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
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/xplore/augmentation/random_crop.py/#L48)
```python
.forward(
   x: th.Tensor
)
```

