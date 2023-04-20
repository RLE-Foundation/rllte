#


### make_dmc_env
[source](https://github.com/RLE-Foundation/Hsuanwu\blob\main\hsuanwu/env/dmc/__init__.py\#L52)
```python
.make_dmc_env(
   env_id: str = 'cartpole_balance', num_envs: int = 1, device: Device = 'cuda',
   resource_files: List = None, img_source: str = None, total_frames: int = None,
   seed: int = 1, visualize_reward: bool = False, from_pixels: bool = True,
   height: int = 84, width: int = 84, camera_id: int = 0, frame_stack: int = 3,
   frame_skip: int = 2, episode_length: int = 1000, environment_kwargs: Dict = None
)
```

---
Build DeepMind Control Suite environments.


**Args**

* **env_id** (str) : Name of environment.
* **num_envs** (int) : Number of parallel environments.
* **device** (Device) : Device (cpu, cuda, ...) on which the code should be run.
* **resource_files** (List) : File path of the resource files.
* **img_source** (str) : Type of the background distractor, supported values: ['color', 'noise', 'images', 'video'].
* **total_frames** (int) : for 'images' or 'video' distractor.
* **seed** (int) : Random seed.
* **visualize_reward** (bool) : True when 'from_pixels' is False, False when 'from_pixels' is True.
* **from_pixels** (bool) : Provide image-based observations or not.
* **height** (int) : Image observation height.
* **width** (int) : Image observation width.
* **camera_id** (int) : Camera id for generating image-based observations.
* **frame_stack** (int) : Number of stacked frames.
* **frame_skip** (int) : Number of action repeat.
* **episode_length** (int) : Maximum length of an episode.
* **environment_kwargs** (Dict) : Other environment arguments.


**Returns**

Environments instance.
