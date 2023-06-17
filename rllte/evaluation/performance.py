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


from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import random
from scipy import stats as sts

from rllte.evaluation.utils import StratifiedBootstrap


class Performance:
    """Evaluate the performance of an algorithm. Based on:
        https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
        scores (np.ndarray): A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m`.
        get_ci (bool): Compute CIs or not.
        method (str): One of `basic`, `percentile`, `bc` (identical to `debiased`,
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
            CIs = self.get_interval_estimates(scores=self.scores, metric=lambda x: _thunk(x, gamma=gamma))
            return _thunk(self.scores, gamma), CIs
        else:
            return _thunk(self.scores, gamma)

    def aggregate_iqm(self) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Computes the interquartile mean across runs and tasks."""

        def _thunk(scores):
            return sts.trim_mean(scores, proportiontocut=0.25, axis=None)

        if self.get_ci:
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
            scores (np.ndarray): A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
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

    def create_performance_profile(
        self, tau_list: Union[List[float], np.ndarray], use_score_distribution: bool = True
    ) -> Tuple[np.ndarray, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """Method for calculating performance profilies.

        Args:
            tau_list (Union[List[float], np.ndarray]): List of 1D numpy array of threshold
                values on which the profile is evaluated.
            use_score_distribution (bool): Whether to report score distributions or average
                score distributions.

        Returns:
            Point and interval estimates of profiles evaluated at all thresholds in 'tau_list'.
        """
        if use_score_distribution:

            def _thunk(scores, tau):
                return np.mean(scores > tau)

        else:

            def _thunk(scores, tau):
                return np.mean(np.mean(scores, axis=0) > tau)

        profile_function = np.vectorize(_thunk, excluded=[0])
        profiles = profile_function(self.scores, tau_list)
        profile_cis = self.get_interval_estimates(scores=self.scores, metric=lambda x: profile_function(x, tau_list))

        return profiles, profile_cis
