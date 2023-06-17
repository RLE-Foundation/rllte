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


from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy import random
from scipy import stats as sts

from rllte.evaluation.utils import StratifiedIndependentBootstrap


class Comparison:
    """Compare the performance between algorithms. Based on:
        https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
        scores_x (np.ndarray): A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m` for algorithm `X`.
        scores_y (np.ndarray): A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m` for algorithm `Y`.
        get_ci (bool): Compute CIs or not.
        method (str): One of `basic`, `percentile`, `bc` (identical to `debiased`,
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
            CIs = self.get_interval_estimates(scores_x=self.scores_x, scores_y=self.scores_y, metric=_thunk)
            return _thunk(self.scores_x, self.scores_y), CIs
        else:
            return _thunk(self.scores_x, self.scores_y)

    def get_interval_estimates(
        self,
        scores_x: np.ndarray,
        scores_y: np.ndarray,
        metric: Callable,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Computes interval estimation of the above performance evaluators.

        Args:
            scores_x (np.ndarray): A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
                represent the score on run `n` of task `m` for algorithm `X`.
            scores_y (np.ndarray): A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
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
