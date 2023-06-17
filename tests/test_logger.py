import os
from pathlib import Path

from rllte.common.logger import Logger

if __name__ == "__main__":
    work_dir = Path.cwd() / "logs/"
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    logger = Logger(work_dir)

    msg_info = "Invoking RLLTE Engine..."
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

    print(logger.time_stamp)
    logger.info(msg="Invoking RLLTE Engine...")
    logger.debug(msg="Checking Module Compatibility...")
    logger.error(msg="Unsupported function!")
    logger.train(msg=msg_train)
    logger.eval(msg=msg_test)

    print("Logger test passed!")
