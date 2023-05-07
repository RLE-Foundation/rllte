#


## ElasticTransform
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/elastic_transform.py/#L7)
```python 
ElasticTransform(
   alpha: float = 50.0, sigma: float = 5.0, interpolation: int = 0, fill = 0
)
```


---
ElasticTransform method based on “ElasticTransform: Transform a image with elastic transformations”.

**Args**

* **alpha** (float or sequence of python:floats) : Magnitude of displacements. Default is 50.0.
* **sigma** (float or sequence of python:floats) : Smoothness of displacements. Default is 5.0.
* **interpolation** (InterpolationMode) : Desired interpolation enum defined by torchvision.transforms.InterpolationMode.
    Default is InterpolationMode.BILINEAR.
* **fill** (sequence or int number) : Pixel fill value for the area outside the transformed image. Default is 0.


**Returns**

Augmented images.


**Methods:**


### .forward
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/xplore/augmentation/elastic_transform.py/#L39)
```python
.forward(
   x: th.Tensor
)
```

