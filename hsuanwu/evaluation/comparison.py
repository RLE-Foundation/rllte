from typing import Any, Tuple, Dict, Callable, Optional

import numpy as np
from numpy import random
from scipy import stats as sts
from hsuanwu.evaluation.utils import StratifiedIndependentBootstrap


class Comparison:
    """Compare the performance between algorithms. Based on:
        https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
        scores_x (NdArray): A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m` for algorithm `X`.
        scores_y (NdArray): A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m` for algorithm `Y`.
        get_ci (bool): Compute CIs or not.
        method (str):  One of `basic`, `percentile`, `bc` (identical to `debiased`,
            `bias-corrected`), or `bca`.
        reps (int): Number of bootstrap replications.
        confidence_interval_size (float): Coverage of confidence interval.
        random_state (int): If specified, ensures reproducibility in uncertainty estimates.

    Returns:
        Comparer instance.
    """

    def __init__(
        self,
        scores_x: np.ndarray,
        scores_y: np.ndarray,
        get_ci: bool = False,
        method: str = "percentile",
        reps: int = 2000,
        confidence_interval_size: float = 0.95,
        random_state: Optional[random.RandomState] = None,
    ) -> None:
        self.scores_x = scores_x
        self.scores_y = scores_y
        assert self.scores_x.shape[1] == self.scores_y.shape[1], "The two scores matrix must have same number of tasks!"
        self.get_ci = get_ci
        self.method = method
        self.reps = reps
        self.confidence_interval_size = confidence_interval_size
        self.random_state = random_state

    def compute_poi(self) -> np.floating:
        """Compute the overall probability of imporvement of algorithm `X` over `Y`."""

        def _thunk(scores_x, scores_y):
            all_ips = list()  # all the imporvement probabilities
            num_tasks = scores_y.shape[1]
            num_runs_x, num_runs_y = scores_x.shape[0], scores_y.shape[0]
            for task in range(num_tasks):
                if np.array_equal(scores_x[:, task], scores_y[:, task]):
                    ip = 0.5
                else:
                    ip, _ = sts.mannwhitneyu(
                        scores_x[:, task],
                        scores_y[:, task],
                        alternative="greater",
                    )
                    ip /= num_runs_x * num_runs_y
                all_ips.append(ip)

            PoI = np.mean(all_ips)
            return PoI

        if self.get_ci:
            print("Computing confidence interval for PoI...")
            CIs = self.get_interval_estimates(scores_x=self.scores_x, scores_y=self.scores_y, metric=_thunk)
            return _thunk(self.scores_x, self.scores_y), CIs
        else:
            return _thunk(self.scores_x, self.scores_y)

    def get_interval_estimates(
        self,
        scores_x: np.array,
        scores_y: np.array,
        metric: Callable,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Computes interval estimation of the above performance evaluators.

        Args:
            scores_x (NdArray): A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
                represent the score on run `n` of task `m` for algorithm `X`.
            scores_y (NdArray): A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
                represent the score on run `n` of task `m` for algorithm `Y`.
            metric (Callable): One of the above performance evaluators used for estimation.

        Returns:
            Confidence intervals.
        """
        stratified_bs = StratifiedIndependentBootstrap(*[scores_x, scores_y], random_state=self.random_state)
        interval_estimates = stratified_bs.conf_int(
            metric, reps=self.reps, size=self.confidence_interval_size, method=self.method
        )
        return interval_estimates
