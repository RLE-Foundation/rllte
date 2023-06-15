import numpy as np

from rllte.evaluation.comparison import Comparison
from rllte.evaluation.performance import Performance

if __name__ == "__main__":
    scores_x = np.random.rand(10, 5)
    scores_y = np.random.rand(10, 5)

    perf = Performance(scores=scores_x)
    perf.aggregate_iqm()
    perf.aggregate_mean()
    perf.aggregate_median()
    perf.aggregate_og()

    comp = Comparison(scores_x=scores_x, scores_y=scores_y)
    ips = comp.compute_poi()

    print("Evaluation test passed!")
