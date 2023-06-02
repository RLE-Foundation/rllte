from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def _decorate_axis(
    ax: plt.axes, wrect: float = 10, hrect: float = 10, ticklabelsize: str = 'large'
) -> plt.axes:
    """Helper function for decorating plots.

    Args:
        ax (axes): The axes object on which the decorations will be applied
        wrect (int): the outward distance of the bottom spine from the plot
        hrect (int): the outward distance of the left spine from the plot
        ticklabelsize (str): the font of the tick label size
        
    Returns:
        Decorated plots.
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))
    return ax


def _annotate_and_decorate_axis(
    ax: plt.axes,
    labelsize: str = 'x-large',
    ticklabelsize: str = 'x-large',
    xticks: List[float] = None,
    xticklabels: List[str] = None,
    yticks: List[float] = None,
    legend: bool = False,
    grid_alpha: float = 0.2,
    legendsize: str = 'x-large',
    xlabel: str = '',
    ylabel: str = '',
    wrect: float = 10,
    hrect: float = 10
) -> plt.axes:
    """Annotates and decorates the plot.
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


def plot_interval_estimates(
    scores_dict: Dict
) -> None:
    """Plots verious metrics of algorithms with confidence intervals.
    Args:
        scores_dict (Dict): The dictionary of various metrics of algorithms,
            an example is shown below:
            # score1: scores of PPO algorithm with the size (`num_runs` x `num_tasks`)
            # score2: scores of PPG algorithm with the size (`num_runs` x `num_tasks`)
            # score1: scores of DrAC algorithm with the size (`num_runs` x `num_tasks`)
            PPO = Performance(score1, True, reps = 200)
            PPG = Performance(score2, True, reps = 200)
            DrAC = Performance(score3, True, reps = 200)
            MEAN1, MEAN_CIs1 = PPO.aggregate_mean()
            MEAN2, MEAN_CIs2 = PPG.aggregate_mean()
            MEAN3, MEAN_CIs3 = DrAC.aggregate_mean()
            MEDIAN1, MEDIAN_CIs1 = PPO.aggregate_median()
            MEDIAN2, MEDIAN_CIs2 = PPG.aggregate_median()
            scores_dict = {
                'MEAN': {
                    'PPO': [MEAN1, MEAN_CIs1],
                    'PPG': [MEAN2, MEAN_CIs2],
                    'DrAC': [MEAN3, MEAN_CIs3]
                    },
                "MEDIAN": {
                    'PPO': [MEDIAN1, MEDIAN_CIs1],
                    'PPG': [MEDIAN2, MEDIAN_CIs2],
                }
            }
    Returns:
        Matplotlib figures
    """
    for metric, algorithms in scores_dict.items():

        fig_height = 0.45 * len(scores_dict[metric])
        fig, ax = plt.subplots(figsize=(5, fig_height))
        color_palette = sns.color_palette('colorblind', n_colors=len(algorithms.keys()))
        colors = dict(zip(list(algorithms.keys()), color_palette))

        for alg_idx, algorithm in enumerate(scores_dict[metric].keys()):

            lower, upper = scores_dict[metric][algorithm][1]
            height = 0.5

            ax.barh(
                y=algorithm,
                width=upper - lower,
                left=lower,
                height=height,
                color=colors[algorithm],
                alpha=0.6,
                label=algorithm,
            )
            ax.vlines(
                x=scores_dict[metric][algorithm][0],
                ymin=alg_idx - (7.9 * height / 16),
                ymax=alg_idx + (7.9 * height / 16),
                label=algorithm,
                color='k',
                alpha=0.5
            )

        _decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(metric, fontsize='xx-large')
        plt.grid(axis='x')
        fig.text(x=0.47, y=-1.2/len(scores_dict[metric]), s='Normalized score', ha='center', fontsize='x-large')


def plot_probability_improvement(
    poi_dict: Dict
) -> None:
    """Plots probability of improvement with confidence intervals.
    Args:
        poi_dict (Dict): The dictionary of probability of improvements
            of different algorithms pairs, an example is shown below:
            input_dict = {'algorithm1': score1,
                        'algorithm2': score2,
                        'algorithm3': score3}
            compute_pairs = [('algorithm1', 'algorithm2'),
                            ('algorithm1', 'algorithm3'),
                            ('algorithm2', 'algorithm3')]
            poi_dict = {} # the dictionary passed into the function
            for pairs in compute_pairs:
                poi, poi_cls = Comparison(
                                    input_dict[pairs[0]],
                                    input_dict[pairs[1]],
                                    True
                                ).compute_poi()
                poi_dict[pairs] = (poi, poi_cls)
    Returns:
        Matplotlib figures
    """
    fig_height = 0.4 * len(poi_dict)
    _, ax = plt.subplots(figsize=(5, fig_height))

    twin_ax = ax.twinx()
    all_algorithm_x, all_algorithm_y = [], []

    for idx, (algorithms, pois) in enumerate(poi_dict.items()):

        lower, upper = pois[1]
        algorithm_x, algorithm_y = algorithms
        all_algorithm_x.append(algorithm_x)
        all_algorithm_y.append(algorithm_y)

        ax.barh(
            y=idx,
            width=upper - lower,
            height=0.8,
            left=lower,
            alpha=0.6,
            label=algorithm_x
        )

        twin_ax.barh(
            y=idx,
            width=upper - lower,
            height=0.8,
            left=lower,
            alpha=0.0,
            label=algorithm_y
        )
        ax.vlines(
            x=pois[0],
            ymin=idx - (7.9 * 0.8 / 16),
            ymax=idx + (7.9 * 0.8 / 16),
            label=algorithms,
            color='k',
            alpha=0.5
        )

    ax = _annotate_and_decorate_axis(
        ax,
        xticks=None,
        yticks=range(len(poi_dict)),
        xlabel='P(X > Y)',
        ylabel='Algorithm X'
    )
    twin_ax = _annotate_and_decorate_axis(
        twin_ax,
        xticks=None,
        yticks=range(len(poi_dict)),
        xlabel='P(X > Y)',
        ylabel='Algorithm Y'
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    twin_ax.spines['top'].set_visible(False)
    twin_ax.spines['right'].set_visible(False)
    twin_ax.spines['left'].set_visible(False)
    twin_ax.set_yticklabels(all_algorithm_y, fontsize='large')
    ax.set_yticklabels(all_algorithm_x, fontsize='large')

    twin_ax.set_ylabel(
        ylabel='Algorithm Y',
        fontweight='bold',
        rotation='horizontal',
        fontsize='x-large'
    )
    ax.set_ylabel(
        ylabel='Algorithm X',
        fontweight='bold',
        rotation='horizontal',
        fontsize='x-large'
    )

    twin_ax.set_yticklabels(all_algorithm_y, fontsize='x-large')
    ax.set_yticklabels(all_algorithm_x, fontsize='x-large')
    ax.tick_params(axis='both', which='major')
    twin_ax.tick_params(axis='both', which='major')
    ax.spines['left'].set_visible(False)
    twin_ax.spines['left'].set_visible(False)
    ax.yaxis.set_label_coords(-0.2, 1.0)
    twin_ax.yaxis.set_label_coords(1 + 0.7 * 0.2, 1.0 + 0.7/len(poi_dict))
    ax.grid(axis='y')
    twin_ax.grid(axis='y')


def plot_performance_profile(
    profile_list: List
) -> None:
    """Plots performance profiles with stratified confidence intervals.
    Args:
        profile_list (List): A list of performance profiles with different tau values
            in different algorithms, an example is shown below:
            x1 # scores of algorithm1 with the size (`num_runs` x `num_tasks`)
            x2 # scores of algorithm2 with the size (`num_runs` x `num_tasks`)
            x3 # scores of algorithm3 with the size (`num_runs` x `num_tasks`)
            tau1 = np.linspace(0,1,20)
            tau2 = np.linspace(0,0.95,20)
            p11, p11cls = Performance(x1,reps=300).create_performance_profile(tau1)
            p21, p21cls = Performance(x2,reps=300).create_performance_profile(tau1)
            p31, p31cls = Performance(x3,reps=300).create_performance_profile(tau1)
            p12, p12cls = Performance(x1,reps=300).create_performance_profile(tau2)
            p22, p22cls = Performance(x2,reps=300).create_performance_profile(tau2)
            p32, p32cls = Performance(x3,reps=300).create_performance_profile(tau2)
            profile_list = [
                (tau1, {
                    'algorithm1': [p11, p11cls],
                    'algorithm2': [p21, p21cls],
                    'algorithm3': [p31, p31cls]}),
                (tau2, {
                    'algorithm1': [p12, p12cls],
                    'algorithm2': [p22, p22cls],
                    'algorithm3': [p32, p32cls]})
            ]
    Returns:
        Matplotlib figures
    """

    for idx, item in enumerate(profile_list):

        color_palette = sns.color_palette('colorblind', n_colors=len(item[1].keys()))
        colors = dict(zip(list(item[1].keys()), color_palette))
        plt.figure(figsize=(5, 3))
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(wspace=1.5)

        for algorithm, value in item[1].items():

            ax.plot(item[0], value[0], color=colors[algorithm], label=algorithm)
            ax.fill_between(item[0], value[1][0], value[1][1], alpha=0.1)

        plt.xlabel('Normamlilzed score (tau)')
        plt.ylabel('Fractions of runs with score > tau')
        plt.title('')
        plt.grid(True)

        _annotate_and_decorate_axis(
            ax,
            xlabel='Normamlilzed score (tau)',
            ylabel='Fractions of runs with score > tau',
            labelsize='large',
            ticklabelsize='large'
        )

        ax.axhline(0.5, ls='--', color='k', alpha=0.4)
        fake_patches = [mpatches.Patch(color=colors[algorithm], alpha=0.7) for algorithm in item[1].keys()]

        legend = fig.legend(
            fake_patches,
            item[1].keys(),
            loc='upper center',
            fancybox=True, ncol=len(item[1].keys()),
            fontsize='x-large',
            bbox_to_anchor=(0.5, 1)
        )


def plot_sample_efficiency_curve(
    sampling_dict: Dict,
    frames: List
) -> None:
    """Plots an aggregate metric with CIs as a function of environment frames.
    Args:
        sampling_dict (Dict): A dictionary of values of CIs in different frames
        frames: The list containing environment frames to mark on the x-axis
        An example is shown below:
            score1 # scores of algorithm1 with the size (`num_runs` x `num_tasks` x `num_frames` )
            score2 # scores of algorithm2 with the size (`num_runs` x `num_tasks` x `num_frames` )
            score3 # scores of algorithm3 with the size (`num_runs` x `num_tasks` x `num_frames` )
            frames = np.array([1, 10, 25, 50, 75, 100, 125, 150, 175, 200]) - 1 # num_frames >= 200
            def create_frames_iqm(scores, frames, reps=500):
                iqm_scores = []
                iqm_ci_lower = []
                iqm_ci_upper = []
                for frame in frames:
                    score = scores[:,:,frame]
                    iqm_score, iqm_ci = Performance(score, get_ci=True, reps=reps).aggregate_iqm()
                    iqm_scores.append(float(iqm_score))
                    iqm_ci_lower.append(float(iqm_ci[0]))
                    iqm_ci_upper.append(float(iqm_ci[1]))
                sampling_list = [iqm_scores, iqm_ci_lower, iqm_ci_upper]
                return sampling_list
            def create_frames_median(scores, frames, reps=500):
                median_scores = []
                median_ci_lower = []
                median_ci_upper = []
                for frame in frames:
                    score = scores[:,:,frame]
                    median_score, median_ci = Performance(score, get_ci=True, reps=reps).aggregate_median()
                    median_scores.append(float(median_score))
                    median_ci_lower.append(float(median_ci[0]))
                    median_ci_upper.append(float(median_ci[1]))
                sampling_list = [median_scores, median_ci_lower, median_ci_upper]
                return sampling_list
            iqm_dict = {}
            iqm_dict['algorithmm1'] = create_frames_iqm(score1, frames)
            iqm_dict['algorithmm2'] = create_frames_iqm(score2, frames)
            iqm_dict['algorithmm3'] = create_frames_iqm(score3, frames)
            median_dict = {}
            median_dict['algorithmm1'] = create_frames_median(score1, frames)
            median_dict['algorithmm2'] = create_frames_median(score2, frames)
            median_dict['algorithmm3'] = create_frames_median(score3, frames)
            sampling_dict = {} # the dictionary passed into the function
            sampling_dict['IQM'] = iqm_dict
            sampling_dict['MEDIAN'] = median_dict
    """
    for idx, (metric, algorithms) in enumerate(sampling_dict.items()):

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(wspace=1.5)
        color_palette = sns.color_palette('colorblind', n_colors=len(algorithms.keys()))
        colors = dict(zip(list(algorithms.keys()), color_palette))

        for algorithm, item in algorithms.items():

            ax.plot(frames, item[0], color=colors[algorithm], label=algorithm, marker='o')
            ax.fill_between(frames, item[1], item[2], alpha=0.1)

        plt.xlabel('Number of frames (in millions)')
        plt.ylabel('{} normalized score'.format(metric))
        plt.grid(True)

        _annotate_and_decorate_axis(
            ax,
            xlabel='Number of frames (in millions)',
            ylabel='{} normalized score'.format(metric)
        )
        plt.title('')

        ax.axhline(0.5, ls='--', color='k', alpha=0.4)
        fake_patches = [mpatches.Patch(color=colors[algorithm], alpha=0.7) for algorithm in algorithms]

        legend = fig.legend(
            fake_patches,
            algorithms,
            loc='upper center',
            fancybox=True,
            ncol=len(algorithms),
            fontsize='x-large',
            bbox_to_anchor=(0.5, 1)
        )