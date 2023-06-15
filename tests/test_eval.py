import os
import sys
import numpy as np

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from rllte.evaluation.performance import Performance
from rllte.evaluation.comparison import Comparison

if __name__ == "__main__":
    scores_x = np.random.rand(10, 5)
    scores_y = np.random.rand(10, 5)

    perf = Performance(scores=scores_x)
    perf.aggregate_iqm()
    perf.aggregate_mean()
    perf.aggregate_median()
    perf.aggregate_og()
    perf.create_performance_profile()

    comp = Comparison(
        scores_x=scores_x,
        scores_y=scores_y
    )
    ips = comp.compute_poi()

    print("Evaluation test passed!")