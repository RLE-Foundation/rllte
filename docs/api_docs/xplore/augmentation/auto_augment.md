#


## AutoAugment
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/auto_augment.py/#L7)
```python 
AutoAugment(
   augment_policy: str = T.AutoAugmentPolicy.IMAGENET
)
```


---
Augmentation method based on “AutoAugment: Learning Augmentation Strategies from Data”.

**Args**

* **augment_policy** (str) : Desired policy enum defined by torchvision.transforms.autoaugment.AutoAugmentPolicy.
    Default is AutoAugmentPolicy.IMAGENET.


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/auto_augment.py/#L24)
```python
.forward(
   x: th.Tensor
)
```

