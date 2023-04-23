from typing import Callable, Dict, List, Optional, Tuple, Union, Mapping
import numpy as np
from numpy import random

from hsuanwu.evaluation.utils import (
    StratifiedBootstrap,
    StratifiedIndependentBootstrap
)
from hsuanwu.evaluation.metrics import (
    score_distributions,
    average_score_distributions
)
Float = Union[float, np.float32, np.float64]


def get_interval_estimates(
    score_dict: Union[Mapping[str, np.ndarray],
                      Mapping[str, List[np.ndarray]]],
    func: Callable[..., np.ndarray],
    method: str = 'percentile',
    task_bootstrap: bool = False,
    reps: int = 50000,
    confidence_interval_size: Float = 0.95,
    random_state: Optional[random.RandomState] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Computes interval estimates by stratified bootstrap confidence intervals.
    Args:
      score_dict: A dictionary of scores for each method where scores are
        arranged as a matrix of the shape (`num_runs` x `num_tasks` x ..).
        For example, the scores could be 2D matrix containing final scores
        of the algorithm or a 3D matrix containing evaluation scores at
        multiple points during training.
      func: Function that computes the aggregate performance, which outputs a
        1D numpy array. See Notes for requirements. For example, if computing
        estimates for interquartile mean across all runs, pass the function as
        `lambda x: np.array([metrics.aggregate_IQM])`.
      method:  One of `basic`, `percentile`, `bc` (identical to `debiased`,
        `bias-corrected’), or ‘bca`.
      task_bootstrap:  Whether to perform bootstrapping over tasks in addition
        to runs. Defaults to False. See `StratifiedBoostrap` for more details.
      reps: Number of bootstrap replications.
      confidence_interval_size: Coverage of confidence interval.
        Defaults to 95%.
      random_state: If specified, ensures reproducibility in uncertainty
        estimates.
    Returns:
      point_estimates: A dictionary of point estimates obtained by applying
        `func` on score data corresponding to each key in `data_dict`.
      interval_estimates: Confidence intervals~(CIs) for point estimates.
        Default is to return 95% CIs. Returns a np array of size (2 x ..) where
        the first row contains the lower bounds while the second row contains
        the upper bound of the 95% CIs.
    Notes:
      When there are no extra keyword arguments, the function is called
      .. code:: python
          func(*args, **kwargs)
      where args and kwargs are the bootstrap version of the data provided
      when setting up the bootstrap.  When extra keyword arguments are used,
      these are appended to kwargs before calling func.
      The bootstraps are:
      * 'basic' - Basic confidence using the estimated parameter and
        difference between the estimated parameter and the bootstrap
        parameters.
      * 'percentile' - Direct use of bootstrap percentiles.
      * 'bc' - Bias corrected using estimate bootstrap bias correction.
      * 'bca' - Bias corrected and accelerated, adding acceleration parameter
        to 'bc' method.
    """
    interval_estimates, point_estimates = {}, {}
    for key, scores in score_dict.items():
        print('Calculating estimates for %s ...', key)
        if isinstance(scores, np.ndarray):
            stratified_bs = StratifiedBootstrap(
                scores,
                task_bootstrap=task_bootstrap,
                random_state=random_state)
            point_estimates[key] = func(scores)
        else:
            # Pass arrays as separate arguments, `task_bootstrap` unsupported
            stratified_bs = StratifiedIndependentBootstrap(
                *scores,
                random_state=random_state)
            point_estimates[key] = func(*scores)
        interval_estimates[key] = stratified_bs.conf_int(
            func, reps=reps, size=confidence_interval_size, method=method)
    return point_estimates, interval_estimates


def create_performance_profile(
    score_dict: Mapping[str, np.ndarray],
    tau_list: Union[List[Float], np.ndarray],
    use_score_distribution: bool = True,
    custom_profile_func: Optional[Callable[..., np.ndarray]] = None,
    method: str = 'percentile',
    task_bootstrap: bool = False,
    reps: int = 2000,
    confidence_interval_size: Float = 0.95
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Function for calculating performance profiles.
    Args:
      score_dict: A dictionary of scores for each method where scores are
        arranged as a matrix of the shape (`num_runs` x `num_tasks` x ..).
      tau_list: List or 1D numpy array of threshold values on which the
        profile is evaluated.
      use_score_distribution: Whether to report score distributions or average
        score distributions. Defaults to score distributions for smaller
        uncertainty in reported results with unbiased profiles.ƒ
      custom_profile_func: Custom performance profile function. Can be used to
        compute performance profiles other than score distributions.
      method: Bootstrap method for `StratifiedBootstrap`, defaults to
        percentile.
      task_bootstrap:  Whether to perform bootstrapping over tasks in addition
        to runs. Defaults to False. See `StratifiedBoostrap` for more details.
      reps: Number of bootstrap replications.
      confidence_interval_size: Coverage of confidence interval.
        Defaults to 95%.
    Returns:
      profiles: A dictionary of performance profiles for each key i
        `score_dict`. Each profile is a 1D np array of same size as `tau_list`.
      profile_cis: The 95% confidence intervals of profiles evaluated at
        all threshdolds in `tau_list`.
    """

    if custom_profile_func is None:

        def profile_function(scores):
            if use_score_distribution:
                # Performance profile for scores across all tasks and runs
                return score_distributions(scores, tau_list)
            # Performance profile for task scores averaged across runs
            return average_score_distributions(scores, tau_list)
    else:
        def profile_function(scores):
            return custom_profile_func(scores, tau_list)

    profiles, profile_cis = get_interval_estimates(
        score_dict,
        func=profile_function,
        task_bootstrap=task_bootstrap,
        method=method,
        reps=reps,
        confidence_interval_size=confidence_interval_size)
    return profiles, profile_cis
