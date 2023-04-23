import os
import sys
import numpy as np

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.evaluation.performance import Performance
from hsuanwu.evaluation.comparison import Comparison

scores_x = np.random.rand(10, 5)
scores_y = np.random.rand(10, 5)

perf = Performance(scores=scores_x)
perf.describe()

comp = Comparison(
    scores_x=scores_x,
    scores_y=scores_y
)
ips = comp.compute_poi()