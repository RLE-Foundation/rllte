#


## RandomPerspective
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_perspective.py/#L7)
```python 
RandomPerspective(
   distortion_scale: float = 0.5, p: float = 0.5, interpolation: int = 0, fill = 0
)
```


---
RandomPerspective method based on “RandomPerspective: Performs
a random perspective transformation of the given image with a given probability.”.

**Args**

* **distortion_scale** (float) : argument to control the degree of distortion and ranges from 0 to 1. Default is 0.5.
* **p** (float) : Smoothness of displacements. Default is 5.0.
* **interpolation** (Union, InterpolationMode) : Desired interpolation enum defined by
    torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR.
* **fill** (sequence or int number) : Pixel fill value for the area outside the transformed image. Default is 0.


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/random_perspective.py/#L40)
```python
.forward(
   x: th.Tensor
)
```

