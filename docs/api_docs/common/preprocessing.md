#


### process_observation_space
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py/#L34)
```python
.process_observation_space(
   observation_space: gym.Space
)
```

---
Process the observation space.


**Args**

* **observation_space** (gym.Space) : Observation space.


**Returns**

Information of the observation space.

----


### process_action_space
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py/#L63)
```python
.process_action_space(
   action_space: gym.Space
)
```

---
Get the dimension of the action space.


**Args**

* **action_space** (gym.Space) : Action space.


**Returns**

Information of the action space.

----


### get_flattened_obs_dim
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py/#L97)
```python
.get_flattened_obs_dim(
   observation_space: spaces.Space
)
```

---
Get the dimension of the observation space when flattened. It does not apply to image observation space.
Borrowed from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py#L169


**Args**

* **observation_space** (spaces.Space) : Observation space.


**Returns**

The dimension of the observation space when flattened.

----


### is_image_space_channels_first
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py/#L114)
```python
.is_image_space_channels_first(
   observation_space: spaces.Box
)
```

---
Check if an image observation space (see ``is_image_space``)
is channels-first (CxHxW, True) or channels-last (HxWxC, False).
Use a heuristic that channel dimension is the smallest of the three.
If second dimension is smallest, raise an exception (no support).

Borrowed from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py#L10


**Args**

* **observation_space** (spaces.Box) : Observation space.


**Returns**

True if observation space is channels-first image, False if channels-last.

----


### is_image_space
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py/#L134)
```python
.is_image_space(
   observation_space: gym.Space, check_channels: bool = False,
   normalized_image: bool = False
)
```

---
Check if a observation space has the shape, limits and dtype of a valid image.
The check is conservative, so that it returns False if there is a doubt.
Valid images: RGB, RGBD, GrayScale with values in [0, 255]

Borrowed from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py#L27


**Args**

* **observation_space** (gym.Space) : Observation space.
* **check_channels** (bool) : Whether to do or not the check for the number of channels.
    e.g., with frame-stacking, the observation space may have more channels than expected.
* **normalized_image** (bool) : Whether to assume that the image is already normalized
    or not (this disables dtype and bounds checks): when True, it only checks that
    the space is a Box and has 3 dimensions.
    Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).


**Returns**

True if observation space is channels-first image, False if channels-last.

----


### preprocess_obs
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py/#L178)
```python
.preprocess_obs(
   obs: th.Tensor, observation_space: gym.Space
)
```

---
Observations preprocessing function.
Borrowed from: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py#L92


**Args**

* **obs** (th.Tensor) : Observation.
* **observation_space** (gym.Space) : Observation space.


**Returns**

A function to preprocess observations.
