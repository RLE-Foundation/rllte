import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from pathlib import Path

from hsuanwu.common.logger import *

work_dir = Path.cwd() / "logs/"
os.mkdir(work_dir)
logger = Logger(work_dir)

msg_info = "Invoking Hsuanwu Engine..."
msg_debug = "Checking Module Compatibility..."
msg_train = {
    "frame": 12000,
    "step": 6000,
    "episode": 12,
    "episode_reward": 225.8908,
    "episode_length": 1000.000,
    "fps": 177.1211132,
    "total_time": 1200,
}
msg_test = {
    "frame": 10000,
    "step": 5000,
    "episode": 10,
    "episode_reward": 201.8908,
    "episode_length": 1000.000,
    "total_time": 1000,
}
logger.log(level=INFO, msg=msg_info)
logger.log(level=DEBUG, msg=msg_debug)
logger.log(level=TRAIN, msg=msg_train)
logger.log(level=TEST, msg=msg_test)
