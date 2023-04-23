import os
import sys
import numpy as np

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.evaluation import (
    get_interval_estimates,
    create_performance_profile,
    load_scores_from_json_atari,
    aggregate_mean,
    aggregate_median,
    aggregate_iqm,
    aggregate_optimality_gap

)


# using OTRainbow as a test
DICT_PATH = '../hsuanwu/evaluation/atari_100k/OTRainbow.json'
DICT_NAME = 'OTR'
ATARI_100K_TAU = np.linspace(0.0, 2.0, 201)
REPS = 2000

# loading score dict from json path
score_dict_otr, score_otr = load_scores_from_json_atari(DICT_PATH)


def aggregate_func(x):
    '''the aggregation of the used metrics'''
    return np.array([aggregate_median(x),
                     aggregate_iqm(x),
                     aggregate_mean(x),
                     aggregate_optimality_gap(x)])


# obtain interval estimations of the four metrics
aggregate_scores, aggregate_interval_estimates = get_interval_estimates(
    {DICT_NAME: score_otr}, aggregate_func, reps=50000)


# obtain score distributions of performance profile results
score_distributions, score_distributions_cis = create_performance_profile(
    {DICT_NAME: score_otr}, ATARI_100K_TAU, reps=REPS)
avg_score_distributions, avg_score_distributions_cis = \
    create_performance_profile(
        {DICT_NAME: score_otr},
        ATARI_100K_TAU,
        use_score_distribution=False,
        reps=REPS
    )
