# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _decorate_axis(ax: plt.axes, wrect: float = 10, hrect: float = 10, ticklabelsize: str = "large") -> plt.axes:
    """Helper function for decorating plots.
        Borrowed from: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py

    Args:
        ax (axes): The axes object on which the decorations will be applied
        wrect (int): the outward distance of the bottom spine from the plot
        hrect (int): the outward distance of the left spine from the plot
        ticklabelsize (str): the font of the tick label size

    Returns:
        Decorated plots.
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax


def _annotate_and_decorate_axis(
    ax: plt.axes,
    labelsize: str = "x-large",
    ticklabelsize: str = "x-large",
    xticks: Optional[Iterable] = None,
    xticklabels: Optional[Iterable] = None,
    yticks: Optional[Iterable] = None,
    legend: bool = False,
    grid_alpha: float = 0.2,
    legendsize: str = "x-large",
    xlabel: str = "",
    ylabel: str = "",
    wrect: float = 10,
    hrect: float = 10,
) -> plt.axes:
    """Annotates and decorates the plot.
        Borrowed from: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py

    Args:
        ax (axes): The axes object on which the annotations and decorations will be applied
        labelsize (str): The font size of the x-axis and y-axis labels.
        wrect (int): the outward distance of the bottom spine from the plot
        hrect (int): the outward distance of the left spine from the plot
        ticklabelsize (str): the font of the tick label size

    Returns:
        Decorated plots.
    """
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(xticklabels)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=grid_alpha)
    ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax


def _non_linear_scaling(
    profile_dict: Dict[str, List],
    tau_list: np.ndarray,
    xticklabels: Optional[List] = None,
    num_points: int = 5,
    log_base: float = 2,
) -> Tuple:
    """Returns non linearly scaled tau as well as corresponding xticks.
        The non-linear scaling of a certain range of threshold values is
        proportional to fraction of runs that lie within that range.
        Borrowed from: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py

    Args:
        profile_dict (Dict[str, List]): A dictionary mapping a method to its performance.
        tau_list (np.ndarray): 1D numpy array of threshold values on which the profile is evaluated.
        xticklabels (List[str]): x-axis labels correspond to non-linearly scaled thresholds.
        num_points (int): If `xticklabels` are not passed, then specifices
            the number of indices to be generated on a log scale.
        log_base (float): Base of the logarithm scale for non-linear scaling.

    Returns:
        nonlinear_tau: Non-linearly scaled threshold values.
        new_xticks: x-axis ticks from `nonlinear_tau` that would be plotted.
        xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
    """
    methods = list(profile_dict.keys())
    nonlinear_tau = np.zeros_like(profile_dict[methods[0]][0])
    for method in methods:
        nonlinear_tau += profile_dict[method][0]
    nonlinear_tau /= len(methods)
    nonlinear_tau = 1 - nonlinear_tau

    if xticklabels is None:
        tau_indices = np.int32(np.logspace(start=0, stop=np.log2(len(tau_list) - 1), base=log_base, num=num_points))
        xticklabels = [tau_list[i] for i in tau_indices]
    else:
        tau_as_list = list(tau_list)
        # Find indices of x which are in `tau`
        tau_indices = [tau_as_list.index(x) for x in xticklabels]
    new_xticks = nonlinear_tau[tau_indices]
    return nonlinear_tau, new_xticks, xticklabels


def plot_interval_estimates(
    metrics_dict: Dict[str, Dict],
    metric_names: List[str],
    algorithms: List[str],
    colors: Optional[List[str]] = None,
    color_palette: str = "colorblind",
    max_ticks: float = 4,
    subfigure_width: float = 3.4,
    row_height: float = 0.37,
    interval_height: float = 0.6,
    xlabel_y_coordinate: float = -0.16,
    xlabel: str = "Normalized Score",
    **kwargs,
) -> Tuple[plt.figure, plt.Axes]:
    """Plots verious metrics of algorithms with stratified confidence intervals.
        Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
        See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.

    Args:
        metrics_dict (Dict[str, Dict]): The dictionary of various metrics of algorithms.
        metric_names (List[str]): Names of the metrics corresponding to `metrics_dict`.
        algorithms (List[str]): List of methods used for plotting.
        colors (Optional[List[str]]): Maps each method to a color.
            If None, then this mapping is created based on `color_palette`.
        color_palette (str): `seaborn.color_palette` object for mapping each method to a color.
        max_ticks (float): Find nice tick locations with no more than `max_ticks`. Passed to `plt.MaxNLocator`.
        subfigure_width (float): Width of each subfigure.
        row_height (float): Height of each row in a subfigure.
        interval_height (float): Height of confidence intervals.
        xlabel_y_coordinate (float): y-coordinate of the x-axis label.
        xlabel (str): Label for the x-axis.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        A matplotlib figure and an array of Axes.
    """
    num_metrics = len(metric_names)
    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
    colors = dict(zip(algorithms, color_palette))

    for idx, metric_name in enumerate(metric_names):
        for algo_idx, algo in enumerate(algorithms):
            ax = axes[idx] if num_metrics > 1 else axes
            # plot interval estimates
            lower, upper = metrics_dict[metric_name][algo][1]

            ax.barh(
                y=algo_idx, width=upper - lower, height=interval_height, left=lower, color=colors[algo], alpha=0.75, label=algo
            )

            ax.vlines(
                x=metrics_dict[metric_name][algo][0],
                ymin=algo_idx - (7.5 * interval_height / 16),
                ymax=algo_idx + (6 * interval_height / 16),
                label=algo,
                color="k",
                alpha=0.5,
            )

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if idx != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(algorithms, fontsize="x-large")
        ax.set_title(metric_name, fontsize="xx-large")
        ax.tick_params(axis="both", which="major")
        _decorate_axis(ax, ticklabelsize="xx-large", wrect=5)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25)

    fig.text(0.4, xlabel_y_coordinate, xlabel, ha="center", fontsize="xx-large")
    plt.subplots_adjust(wspace=kwargs.pop("wspace", 0.11), left=0.0)
    return fig, axes


def plot_probability_improvement(
    poi_dict: Dict[str, List],
    pair_separator: str = "_",
    figsize: Tuple[float, float] = (3.7, 2.1),
    colors: Optional[List[str]] = None,
    color_palette: str = "colorblind",
    alpha: float = 0.75,
    interval_height: float = 0.6,
    xticks: Optional[Iterable] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    xlabel: str = "P(X > Y)",
    left_ylabel: str = "Algorithm X",
    right_ylabel: str = "Algorithm Y",
    **kwargs,
) -> Tuple[plt.figure, plt.Axes]:
    """Plots probability of improvement with stratified confidence intervals.
        Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
        See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.

    Args:
        poi_dict (Dict[str, List]): The dictionary of probability of improvements of different algorithms pairs.
        pair_separator (str): Each algorithm pair name in dictionaries above is joined by a string separator.
            For example, if the pairs are specified as 'X;Y', then the separator corresponds to ';'. Defaults to ','.
        figsize (Tuple[float]): Size of the figure passed to `matplotlib.subplots`.
        colors (Optional[List[str]]): Maps each method to a color. If None, then this mapping
            is created based on `color_palette`.
        color_palette (str): `seaborn.color_palette` object for mapping each method to a color.
        interval_height (float): Height of confidence intervals.
        alpha (float): Changes the transparency of the shaded regions corresponding to the confidence intervals.
        xticks (Optional[Iterable]): The list of x-axis tick locations. Passing an empty list removes all xticks.
        xlabel (str): Label for the x-axis.
        left_ylabel (str): Label for the left y-axis. Defaults to 'Algorithm X'.
        right_ylabel (str): Label for the left y-axis. Defaults to 'Algorithm Y'.
        **kwargs: Arbitrary keyword arguments for annotating and decorating the
            figure. For valid arguments, refer to `_annotate_and_decorate_axis`.

    Returns:
        A matplotlib figure and `axes.Axes` which contains the plot for probability of improvement.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if not colors:
        colors = sns.color_palette(color_palette, n_colors=len(poi_dict.keys()))
    wrect = kwargs.pop("wrect", 5)
    ticklabelsize = kwargs.pop("ticklabelsize", "x-large")
    labelsize = kwargs.pop("labelsize", "x-large")
    # x-position of the y-label
    ylabel_x_coordinate = kwargs.pop("ylabel_x_coordinate", 0.2)
    # x-position of the y-label
    twin_ax = ax.twinx()
    all_algorithm_x, all_algorithm_y = [], []

    for idx, (algorithm_pair, pois) in enumerate(poi_dict.items()):
        lower, upper = pois[1]
        algorithm_x, algorithm_y = algorithm_pair.split(pair_separator)
        all_algorithm_x.append(algorithm_x)
        all_algorithm_y.append(algorithm_y)

        ax.barh(y=idx, width=upper - lower, height=interval_height, left=lower, alpha=alpha, label=algorithm_x)

        twin_ax.barh(y=idx, width=upper - lower, height=interval_height, left=lower, alpha=0.0, label=algorithm_y)
        ax.vlines(
            x=pois[0],
            ymin=idx - 7.5 * interval_height / 16,
            ymax=idx + (6 * interval_height / 16),
            color="k",
            alpha=min(alpha + 0.1, 1.0),
        )

    yticks = range(len(poi_dict.keys()))
    ax = _annotate_and_decorate_axis(
        ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticks,
        xlabel=xlabel,
        ylabel=left_ylabel,
        wrect=wrect,
        ticklabelsize=ticklabelsize,
        labelsize=labelsize,
        **kwargs,
    )
    twin_ax = _annotate_and_decorate_axis(
        twin_ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticks,
        xlabel=xlabel,
        ylabel=right_ylabel,
        wrect=wrect,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        grid_alpha=0.0,
        **kwargs,
    )
    twin_ax.set_yticklabels(all_algorithm_y, fontsize="large")
    ax.set_yticklabels(all_algorithm_x, fontsize="large")
    twin_ax.set_ylabel(right_ylabel, fontweight="bold", rotation="horizontal", fontsize=labelsize)
    ax.set_ylabel(left_ylabel, fontweight="bold", rotation="horizontal", fontsize=labelsize)
    twin_ax.set_yticklabels(all_algorithm_y, fontsize=ticklabelsize)
    ax.set_yticklabels(all_algorithm_x, fontsize=ticklabelsize)
    ax.tick_params(axis="both", which="major")
    twin_ax.tick_params(axis="both", which="major")
    ax.spines["left"].set_visible(False)
    twin_ax.spines["left"].set_visible(False)
    plt.subplots_adjust(wspace=0.05)
    ax.yaxis.set_label_coords(-ylabel_x_coordinate, 1.0)
    twin_ax.yaxis.set_label_coords(1 + 0.7 * ylabel_x_coordinate, 1 + 0.6 * ylabel_x_coordinate)

    return fig, ax


def plot_performance_profile(
    profile_dict: Dict[str, List],
    tau_list: np.ndarray,
    use_non_linear_scaling: bool = False,
    figsize: Tuple[float, float] = (10.0, 5.0),
    colors: Optional[List[str]] = None,
    color_palette: str = "colorblind",
    alpha: float = 0.15,
    xticks: Optional[Iterable] = None,
    yticks: Optional[Iterable] = None,
    xlabel: Optional[str] = r"Normalized Score ($\tau$)",
    ylabel: Optional[str] = r"Fraction of runs with score $> \tau$",
    linestyles: Optional[str] = None,
    **kwargs,
) -> Tuple[plt.figure, plt.Axes]:
    """Plots performance profiles with stratified confidence intervals.
        Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
        See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.

    Args:
        profile_dict (Dict[str, List]): A dictionary mapping a method to its performance.
        tau_list (np.ndarray): 1D numpy array of threshold values on which the profile is evaluated.
        use_non_linear_scaling (bool): Whether to scale the x-axis in proportion to the
            number of runs within any specified range.
        figsize (Tuple[float]): Size of the figure passed to `matplotlib.subplots`.
        colors (Optional[List[str]]): Maps each method to a color. If None, then
            this mapping is created based on `color_palette`.
        color_palette (str): `seaborn.color_palette` object for mapping each method to a color.
        alpha (float): Changes the transparency of the shaded regions corresponding to the confidence intervals.
        xticks (Optional[Iterable]): The list of x-axis tick locations. Passing an empty list removes all xticks.
        yticks (Optional[Iterable]): The list of y-axis tick locations between 0 and 1.
            If None, defaults to `[0, 0.25, 0.5, 0.75, 1.0]`.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        linestyles (str): Maps each method to a linestyle. If None, then the 'solid' linestyle is used for all methods.
        **kwargs: Arbitrary keyword arguments for annotating and decorating the
            figure. For valid arguments, refer to `_annotate_and_decorate_axis`.

    Returns:
        A matplotlib figure and `axes.Axes` which contains the plot for performance profiles.
    """
    legend = kwargs.pop("legend", True)

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        keys = profile_dict.keys()
        color_palette = sns.color_palette(color_palette, n_colors=len(keys))
        colors = dict(zip(list(keys), color_palette))

    if linestyles is None:
        linestyles = {key: "solid" for key in profile_dict.keys()}

    if use_non_linear_scaling:
        tau_list, xticks, xticklabels = _non_linear_scaling(profile_dict, tau_list, xticks)
    else:
        xticklabels = xticks

    for method, profile in profile_dict.items():
        ax.plot(
            tau_list,
            profile[0],
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=kwargs.pop("linewidth", 2.0),
            label=method,
        )

        lower_ci, upper_ci = profile[1]
        ax.fill_between(tau_list, lower_ci, upper_ci, color=colors[method], alpha=alpha)

    if yticks is None:
        yticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    return fig, _annotate_and_decorate_axis(
        ax, xticks=xticks, yticks=yticks, xticklabels=xticklabels, xlabel=xlabel, ylabel=ylabel, legend=legend, **kwargs
    )


def plot_sample_efficiency_curve(
    sampling_dict: Dict[str, Dict],
    frames: np.ndarray,
    algorithms: List[str],
    colors: Optional[List[str]] = None,
    color_palette: str = "colorblind",
    figsize: Tuple[float, float] = (3.7, 2.1),
    xlabel: Optional[str] = r"Number of Frames (in millions)",
    ylabel: Optional[str] = r"Aggregate Human Normalized Score",
    labelsize: str = "xx-large",
    ticklabelsize: str = "xx-large",
    **kwargs,
) -> Tuple[plt.figure, plt.Axes]:
    """Plots an aggregate metric with CIs as a function of environment frames.
        Based on: https://github.com/google-research/rliable/blob/master/rliable/plot_utils.py
        See https://docs.rllte.dev/tutorials/evaluation/ for usage tutorials.

    Args:
        sampling_dict (Dict[str, Dict]): A dictionary of values with stratified confidence intervals in different frames.
        frames (np.ndarray): Array containing environment frames to mark on the x-axis.
        algorithms (List[str]): List of methods used for plotting.
        colors (Optional[List[str]]): Maps each method to a color. If None, then this mapping
            is created based on `color_palette`.
        color_palette (str): `seaborn.color_palette` object for mapping each method to a color.
        max_ticks (float): Find nice tick locations with no more than `max_ticks`. Passed to `plt.MaxNLocator`.
        subfigure_width (float): Width of each subfigure.
        row_height (float): Height of each row in a subfigure.
        interval_height (float): Height of confidence intervals.
        xlabel_y_coordinate (float): y-coordinate of the x-axis label.
        xlabel (str): Label for the x-axis.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        A matplotlib figure and an array of Axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
    colors = dict(zip(algorithms, color_palette))

    for algorithm in algorithms:
        metric_values = sampling_dict[algorithm][0]
        lower, upper = sampling_dict[algorithm][1], sampling_dict[algorithm][2]
        ax.plot(
            frames,
            metric_values,
            color=colors[algorithm],
            marker=kwargs.get("marker", "o"),
            linewidth=kwargs.get("linewidth", 2),
            label=algorithm,
        )
        ax.fill_between(frames, y1=lower, y2=upper, color=colors[algorithm], alpha=0.2)
        kwargs.pop("marker", "0")
        kwargs.pop("linewidth", "2")

    return fig, _annotate_and_decorate_axis(
        ax, xlabel=xlabel, ylabel=ylabel, labelsize=labelsize, ticklabelsize=ticklabelsize, **kwargs
    )
