import json
import numpy as np

from hsuanwu.evaluation.metrics import aggregate_mean, aggregate_median

ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]

RANDOM_SCORES = {
 'Alien': 227.8,
 'Amidar': 5.8,
 'Assault': 222.4,
 'Asterix': 210.0,
 'BankHeist': 14.2,
 'BattleZone': 2360.0,
 'Boxing': 0.1,
 'Breakout': 1.7,
 'ChopperCommand': 811.0,
 'CrazyClimber': 10780.5,
 'DemonAttack': 152.1,
 'Freeway': 0.0,
 'Frostbite': 65.2,
 'Gopher': 257.6,
 'Hero': 1027.0,
 'Jamesbond': 29.0,
 'Kangaroo': 52.0,
 'Krull': 1598.0,
 'KungFuMaster': 258.5,
 'MsPacman': 307.3,
 'Pong': -20.7,
 'PrivateEye': 24.9,
 'Qbert': 163.9,
 'RoadRunner': 11.5,
 'Seaquest': 68.4,
 'UpNDown': 533.4
}

HUMAN_SCORES = {
 'Alien': 7127.7,
 'Amidar': 1719.5,
 'Assault': 742.0,
 'Asterix': 8503.3,
 'BankHeist': 753.1,
 'BattleZone': 37187.5,
 'Boxing': 12.1,
 'Breakout': 30.5,
 'ChopperCommand': 7387.8,
 'CrazyClimber': 35829.4,
 'DemonAttack': 1971.0,
 'Freeway': 29.6,
 'Frostbite': 4334.7,
 'Gopher': 2412.5,
 'Hero': 30826.4,
 'Jamesbond': 302.8,
 'Kangaroo': 3035.0,
 'Krull': 2665.5,
 'KungFuMaster': 22736.3,
 'MsPacman': 6951.6,
 'Pong': 14.6,
 'PrivateEye': 69571.3,
 'Qbert': 13455.0,
 'RoadRunner': 7845.0,
 'Seaquest': 42054.7,
 'UpNDown': 11693.2
}


def score_normalization_atari(res_dict, min_scores, max_scores) -> dict:
    '''
    Helper function for normalizing scores using HUMAN_SCORES
    and RANDOM_SCORES.
    Target task: Atari games.
    Args:
        res_dict (dict): Original score dictionary.
        min_scores (int): Lower bound of the scores.
        max_scores (int): Upper bound of the scores.

    Returns:
        Normalized score dictionary.
    '''
    norm_scores = {}
    for game, scores in res_dict.items():
        norm_scores[game] = \
          (scores - min_scores[game])/(max_scores[game] - min_scores[game])
    return norm_scores


def convert_to_matrix(score_dict) -> np.ndarray:
    '''
    Helper function for converting score dictionary to numpy matrix.
    Args:
        score_dict (dict): Score dictionary.

    Returns:
        Numpy instance of score dictionary for analysis and plotting.
    '''
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


def load_scores_from_json_atari(json_path) -> tuple([dict, np.ndarray]):
    '''
    Helper function for loading score dictionaries from a json path
    Target task: Atari games.
    Args:
        json_path (str): Path of the json score dictionary.

    Returns:
        Dict instance of score dictionary.
        Numpy instance of score dictionary.
    '''
    print(f'Loading scores for {json_path}:')
    with open(json_path, 'r', encoding="utf8") as json_scores:
        scores = json.load(json_scores)
    scores = {game: np.array(val) for game, val in scores.items()}
    scores = score_normalization_atari(scores, RANDOM_SCORES, HUMAN_SCORES)
    score_matrix = convert_to_matrix(scores)
    median, mean = aggregate_median(score_matrix), aggregate_mean(score_matrix)
    print('{0}: Median: {1}, Mean: {2}'.format(eval, median, mean))
    return scores, score_matrix
