import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path
from hsuanwu.common.logger import Logger

logger = Logger(Path.cwd() / 'logs/')
metric = {
    'E': 'hopper_hop', # env
    'S': '10000000', # step
    'R': '102.2055', # episode reward,
    'F': '177.0843', # FPS
    'T': '0:00:29' # time cost
}
logger.log(metric)
logger.log(metric, mode='eval')
logger.log(metric)
logger.log(metric, mode='eval')
logger.log(metric)
logger.log(metric, mode='eval')