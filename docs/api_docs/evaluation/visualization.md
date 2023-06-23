#


### plot_interval_estimates
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/visualization.py/#L141)
```python
.plot_interval_estimates(
   metrics_dict: Dict[str, Dict], metric_names: List[str], algorithms: List[str],
   colors: Optional[List[str]] = None, color_palette: str = 'colorblind',
   max_ticks: float = 4, subfigure_width: float = 3.4, row_height: float = 0.37,
   interval_height: float = 0.6, xlabel_y_coordinate: float = -0.16,
   xlabel: str = 'NormalizedScore', **kwargs
)
```

---
Plots verious metrics of algorithms with stratified confidence intervals.
Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.


**Args**

* **metrics_dict** (Dict[str, Dict]) : The dictionary of various metrics of algorithms.
* **metric_names** (List[str]) : Names of the metrics corresponding to `metrics_dict`.
* **algorithms** (List[str]) : List of methods used for plotting.
* **colors** (Optional[List[str]]) : Maps each method to a color.
    If None, then this mapping is created based on `color_palette`.
* **color_palette** (str) : `seaborn.color_palette` object for mapping each method to a color.
* **max_ticks** (float) : Find nice tick locations with no more than `max_ticks`. Passed to `plt.MaxNLocator`.
* **subfigure_width** (float) : Width of each subfigure.
* **row_height** (float) : Height of each row in a subfigure.
* **interval_height** (float) : Height of confidence intervals.
* **xlabel_y_coordinate** (float) : y-coordinate of the x-axis label.
* **xlabel** (str) : Label for the x-axis.
* **kwargs**  : Arbitrary keyword arguments.


**Returns**

A matplotlib figure and an array of Axes.

----


### plot_performance_profile
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/visualization.py/#L331)
```python
.plot_performance_profile(
   profile_dict: Dict[str, List], tau_list: np.ndarray,
   use_non_linear_scaling: bool = False, figsize: Tuple[float, float] = (10.0, 5.0),
   colors: Optional[List[str]] = None, color_palette: str = 'colorblind',
   alpha: float = 0.15, xticks: Optional[Iterable] = None,
   yticks: Optional[Iterable] = None,
   xlabel: Optional[str] = 'NormalizedScore($\\tau$)',
   ylabel: Optional[str] = 'Fractionofrunswithscore$>\\tau$',
   linestyles: Optional[str] = None, **kwargs
)
```

---
Plots performance profiles with stratified confidence intervals.
Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.


**Args**

* **profile_dict** (Dict[str, List]) : A dictionary mapping a method to its performance.
* **tau_list** (np.ndarray) : 1D numpy array of threshold values on which the profile is evaluated.
* **use_non_linear_scaling** (bool) : Whether to scale the x-axis in proportion to the
    number of runs within any specified range.
* **figsize** (Tuple[float]) : Size of the figure passed to `matplotlib.subplots`.
* **colors** (Optional[List[str]]) : Maps each method to a color. If None, then
    this mapping is created based on `color_palette`.
* **color_palette** (str) : `seaborn.color_palette` object for mapping each method to a color.
* **alpha** (float) : Changes the transparency of the shaded regions corresponding to the confidence intervals.
* **xticks** (Optional[Iterable]) : The list of x-axis tick locations. Passing an empty list removes all xticks.
* **yticks** (Optional[Iterable]) : The list of y-axis tick locations between 0 and 1.
    If None, defaults to `[0, 0.25, 0.5, 0.75, 1.0]`.
* **xlabel** (str) : Label for the x-axis.
* **ylabel** (str) : Label for the y-axis.
* **linestyles** (str) : Maps each method to a linestyle. If None, then the 'solid' linestyle is used for all methods.
* **kwargs**  : Arbitrary keyword arguments for annotating and decorating the
    figure. For valid arguments, refer to `_annotate_and_decorate_axis`.


**Returns**

A matplotlib figure and `axes.Axes` which contains the plot for performance profiles.

----


### plot_probability_improvement
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/visualization.py/#L221)
```python
.plot_probability_improvement(
   poi_dict: Dict[str, List], pair_separator: str = '_', figsize: Tuple[float,
   float] = (3.7, 2.1), colors: Optional[List[str]] = None,
   color_palette: str = 'colorblind', alpha: float = 0.75, interval_height: float = 0.6,
   xticks: Optional[Iterable] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
   xlabel: str = 'P(X>Y)', left_ylabel: str = 'AlgorithmX',
   right_ylabel: str = 'AlgorithmY', **kwargs
)
```

---
Plots probability of improvement with stratified confidence intervals.
Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.


**Args**

* **poi_dict** (Dict[str, List]) : The dictionary of probability of improvements of different algorithms pairs.
* **pair_separator** (str) : Each algorithm pair name in dictionaries above is joined by a string separator.
    For example, if the pairs are specified as 'X;Y', then the separator corresponds to ';'. Defaults to ','.
* **figsize** (Tuple[float]) : Size of the figure passed to `matplotlib.subplots`.
* **colors** (Optional[List[str]]) : Maps each method to a color. If None, then this mapping
    is created based on `color_palette`.
* **color_palette** (str) : `seaborn.color_palette` object for mapping each method to a color.
* **interval_height** (float) : Height of confidence intervals.
* **alpha** (float) : Changes the transparency of the shaded regions corresponding to the confidence intervals.
* **xticks** (Optional[Iterable]) : The list of x-axis tick locations. Passing an empty list removes all xticks.
* **xlabel** (str) : Label for the x-axis.
* **left_ylabel** (str) : Label for the left y-axis. Defaults to 'Algorithm X'.
* **right_ylabel** (str) : Label for the left y-axis. Defaults to 'Algorithm Y'.
* **kwargs**  : Arbitrary keyword arguments for annotating and decorating the
    figure. For valid arguments, refer to `_annotate_and_decorate_axis`.


**Returns**

A matplotlib figure and `axes.Axes` which contains the plot for probability of improvement.

----


### plot_sample_efficiency_curve
[source](https://github.com/RLE-Foundation/rllte/blob/main/rllte/evaluation/visualization.py/#L409)
```python
.plot_sample_efficiency_curve(
   sampling_dict: Dict[str, Dict], frames: np.ndarray, algorithms: List[str],
   colors: Optional[List[str]] = None, color_palette: str = 'colorblind',
   figsize: Tuple[float, float] = (3.7, 2.1),
   xlabel: Optional[str] = 'NumberofFrames(inmillions)',
   ylabel: Optional[str] = 'AggregateHumanNormalizedScore',
   labelsize: str = 'xx-large', ticklabelsize: str = 'xx-large', **kwargs
)
```

---
Plots an aggregate metric with CIs as a function of environment frames.
Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.


**Args**

* **sampling_dict** (Dict[str, Dict]) : A dictionary of values with stratified confidence intervals in different frames.
* **frames** (np.ndarray) : Array containing environment frames to mark on the x-axis.
* **algorithms** (List[str]) : List of methods used for plotting.
* **colors** (Optional[List[str]]) : Maps each method to a color. If None, then this mapping
    is created based on `color_palette`.
* **color_palette** (str) : `seaborn.color_palette` object for mapping each method to a color.
* **max_ticks** (float) : Find nice tick locations with no more than `max_ticks`. Passed to `plt.MaxNLocator`.
* **subfigure_width** (float) : Width of each subfigure.
* **row_height** (float) : Height of each row in a subfigure.
* **interval_height** (float) : Height of confidence intervals.
* **xlabel_y_coordinate** (float) : y-coordinate of the x-axis label.
* **xlabel** (str) : Label for the x-axis.
* **kwargs**  : Arbitrary keyword arguments.


**Returns**

A matplotlib figure and an array of Axes.
