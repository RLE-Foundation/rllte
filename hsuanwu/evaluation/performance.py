from typing import Dict

import numpy as np
from scipy import stats as sts


class Performance(object):
    """Evaluate the performance of an algorithm. Based on:
        https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
        scores (NdArray): A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m`.

    Returns:
        Performance evaluator.
    """

    def __init__(
        self,
        scores: np.ndarray,
    ) -> None:
        self.scores = scores
        self.num_runs = scores.shape[0]
        self.num_tasks = scores.shape[1]

    def agg_mean(self) -> np.ndarray:
        """Computes mean of sample mean scores per task."""
        mean_task_scores = np.mean(self.scores, axis=0, keepdims=False)
        return np.mean(mean_task_scores, axis=0)

    def agg_median(self) -> np.ndarray:
        """Computes median of sample mean scores per task."""
        mean_task_scores = np.mean(self.scores, axis=0, keepdims=False)
        return np.median(mean_task_scores, axis=0)

    def agg_og(self, gamma: float = 1.0) -> np.ndarray:
        """Computes optimality gap across all runs and tasks.

        Args:
            gamma (float): Threshold for optimality gap. All scores above `gamma` are clipped
            to `gamma`.

        Returns:
            Optimality gap at threshold `gamma`.
        """
        return gamma - np.mean(np.minimum(self.scores, gamma))

    def agg_iqm(self) -> np.ndarray:
        """Computes the interquartile mean across runs and tasks."""
        return sts.trim_mean(self.scores, proportiontocut=0.25, axis=None)

    def describe(self) -> Dict[str, np.ndarray]:
        """Compute all the evaluation metrics."""
        print(f"Evaluated with {self.num_runs} runs and {self.num_tasks} tasks")
        mean = self.agg_mean()
        median = self.agg_median()
        og = self.agg_og()
        iqm = self.agg_iqm()

        print(f"Aggregate Mean: {mean}")
        print(f"Aggregate Median: {median}")
        print(f"Aggregate Optimality Gap: {og}")
        print(f"Aggregate Interquartile Mean: {iqm}")

        return {"agg_mean": mean, "agg_median": median, "agg_og": og, "agg_iqm": iqm}
