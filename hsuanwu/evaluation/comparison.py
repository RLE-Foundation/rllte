from typing import Dict
import numpy as np
from scipy import stats as sts

class Comparison(object):
    """Compare the performance between algorithms. Based on: 
        https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
        scores_x (NdArray): A matrix of size (`num_runs_x` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m` for algorithm `X`.
        scores_y (NdArray): A matrix of size (`num_runs_y` x `num_tasks`) where scores[n][m]
            represent the score on run `n` of task `m` for algorithm `Y`.

    Returns:
        Comparer instance.
    """
    def __init__(self,
                 scores_x: np.ndarray,
                 scores_y: np.ndarray,
                 ) -> None:
        self.scores_x = scores_x
        self.scores_y = scores_y
        assert self.scores_x.shape[1] == self.scores_y.shape[1], "The two scores matrix must have same number of tasks!"
        self.num_runs_x = scores_x.shape[0]
        self.num_runs_y = scores_y.shape[0]
        self.num_tasks = scores_y.shape[1]
    
    def compute_poi(self) -> np.ndarray:
        """Compute the overall Probability of imporvement of algorithm `X` over `Y`.
        """
        all_ips = list() # all the imporvement probabilities
        for task in range(self.num_tasks):
            if np.array_equal(self.scores_x[:, task], self.scores_y[:, task]):
                ip = 0.5
            else:
                ip, _ = sts.mannwhitneyu(self.scores_x[:, task], self.scores_y[:, task], alternative='greater')
                ip /= (self.num_runs_x * self.num_runs_y)
            all_ips.append(ip)
        
        PoI = np.mean(all_ips)
        print(f"Probability of Imporvement: {PoI}")
        return PoI