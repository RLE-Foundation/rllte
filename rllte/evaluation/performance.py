from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy import random
from scipy import stats as sts

from hsuanwu.evaluation.utils import StratifiedBootstrap


class Performance:
    """Evaluate the performance of an algorithm. Based on:
        https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
        scores (NdArray): A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m`.
        get_ci (bool): Compute CIs or not.
        method (str):  One of `basic`, `percentile`, `bc` (identical to `debiased`,
            `bias-corrected`), or `bca`.
        task_bootstrap (bool):  Whether to perform bootstrapping over tasks in addition to
            runs. Defaults to False. See `StratifiedBoostrap` for more details.
        reps (int): Number of bootstrap replications.
        confidence_interval_size (float): Coverage of confidence interval.
        random_state (int): If specified, ensures reproducibility in uncertainty estimates.

    Returns:
        Performance evaluator.
    """

    def __init__(
        self,
        scores: np.ndarray,
        get_ci: bool = False,
        method: str = "percentile",
        task_bootstrap: bool = False,
        reps: int = 50000,
        confidence_interval_size: float = 0.95,
        random_state: Optional[random.RandomState] = None,
    ) -> None:
        self.scores = scores
        self.get_ci = get_ci
        self.method = method
        self.task_bootstrap = task_bootstrap
        self.reps = reps
        self.confidence_interval_size = confidence_interval_size
        self.random_state = random_state

    def aggregate_mean(self) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Computes mean of sample mean scores per task."""

        def _thunk(scores):
            mean_task_scores = np.mean(scores, axis=0, keepdims=False)
            return np.mean(mean_task_scores, axis=0)

        if self.get_ci:
            print("Computing confidence interval for aggregate MEAN...")
            CIs = self.get_interval_estimates(scores=self.scores, metric=_thunk)
            return _thunk(self.scores), CIs
        else:
            return _thunk(self.scores)

    def aggregate_median(self) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Computes median of sample mean scores per task."""

        def _thunk(scores):
            mean_task_scores = np.mean(scores, axis=0, keepdims=False)
            return np.median(mean_task_scores, axis=0)

        if self.get_ci:
            print("Computing confidence interval for aggregate MEDIAN...")
            CIs = self.get_interval_estimates(scores=self.scores, metric=_thunk)
            return _thunk(self.scores), CIs
        else:
            return _thunk(self.scores)

    def aggregate_og(self, gamma: float = 1.0) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Computes optimality gap across all runs and tasks.

        Args:
            gamma (float): Threshold for optimality gap. All scores above `gamma` are clipped
            to `gamma`.

        Returns:
            Optimality gap at threshold `gamma`.
        """

        def _thunk(scores, gamma):
            return gamma - np.mean(np.minimum(scores, gamma))

        if self.get_ci:
            print("Computing confidence interval for aggregate OG...")
            CIs = self.get_interval_estimates(scores=self.scores, metric=lambda x: _thunk(x, gamma=gamma))
            return _thunk(self.scores, gamma), CIs
        else:
            return _thunk(self.scores, gamma)

    def aggregate_iqm(self) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Computes the interquartile mean across runs and tasks."""

        def _thunk(scores):
            return sts.trim_mean(scores, proportiontocut=0.25, axis=None)

        if self.get_ci:
            print("Computing confidence interval for aggregate IQM...")
            CIs = self.get_interval_estimates(scores=self.scores, metric=_thunk)
            return _thunk(self.scores), CIs
        else:
            return _thunk(self.scores)

    def get_interval_estimates(
        self,
        scores: np.ndarray,
        metric: Callable,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Computes interval estimation of the above performance evaluators.

        Args:
            scores (NdArray): A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
                represent the score on run `n` of task `m`.
            metric (Callable): One of the above performance evaluators used for estimation.

        Returns:
            Confidence intervals.
        """
        stratified_bs = StratifiedBootstrap(scores, task_bootstrap=self.task_bootstrap, random_state=self.random_state)
        interval_estimates = stratified_bs.conf_int(
            metric, reps=self.reps, size=self.confidence_interval_size, method=self.method
        )
        return interval_estimates
