#


## RandomConvolution
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_convolution.py/#L7)
```python 

```


---
Random Convolution operation for image augmentation.


**Args**




**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_convolution.py/#L23)
```python
.forward(
   imgs
)
```

---
random covolution in "network randomization"
(imgs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
