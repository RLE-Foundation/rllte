#


### make_dmc_env
[source](https://github.com/RLE-Foundation/Hsuanwu/blob/main/hsuanwu/env/dmc/__init__.py/#L39)
```python
.make_dmc_env(
   env_id: str = 'cartpole_balance', resource_files: str = None, img_source: str = None,
   total_frames: int = None, seed: int = 1, visualize_reward: bool = True,
   from_pixels: bool = False, height: int = 84, width: int = 84, camera_id: int = 0,
   frame_stack: int = 3, frame_skip: int = 1, episode_length: int = 1000,
   environment_kwargs: Dict = None
)
```

---
Build DeepMind Control Suite environments.


**Args**

* **env_id**  : Name of environment.
* **resource_files**  : File path of the resource files.
* **img_source**  : Type of the background distractor, supported values: ['color', 'noise', 'images', 'video'].
* **total_frames**  : for 'images' or 'video' distractor.
* **seed**  : Random seed.
* **visualize_reward**  : True when 'from_pixels' is False, False when 'from_pixels' is True.
* **from_pixels**  : Provide image-based observations or not.
* **height**  : Image observation height.
* **width**  : Image observation width.
* **camera_id**  : Camera id for generating image-based observations.
* **frame_stack**  : Number of stacked frames.
* **frame_skip**  : Number of action repeat.
* **episode_length**  : Maximum length of an episode.
* **environment_kwargs**  : Other environment arguments.


**Returns**

Environments instance.
