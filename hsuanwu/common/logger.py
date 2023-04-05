import csv
import datetime
import os

from hsuanwu.common.logging import *
from hsuanwu.common.typing import *

TRAIN_MSG_FORMAT = [
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("fps", "FPS", "float"),
    ("total_time", "T", "time"),
]

TEST_MSG_FORMAT = [
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]


class Logger:
    """The logger class.

    Args:
        log_dir: The logging location.

    Returns:
        Logger instance.
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._logger = getLogger(name="Hsuanwu Logger")
        self._logger.setLevel(DEBUG)

        sh = StreamHandler()
        formatter = Formatter(
            "[%(asctime)s] - [%(levelname)s] - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        sh.setFormatter(formatter)
        self._logger.addHandler(sh)

        self._train_file = self._log_dir / "train.log"
        self._test_file = self._log_dir / "test.log"
        self._train_file_write_header = True
        self._test_file_write_header = True

    def _format(self, key: str, value: Any, ty: Type):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.03f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise f"invalid format type: {ty}"

    def parse_train_msg(self, msg: Any):
        pieces = []
        for key, disp_key, ty in TRAIN_MSG_FORMAT:
            value = msg.get(key, 0)
            pieces.append(self._format(disp_key, value, ty).ljust(13, " "))
        return " | ".join(pieces)

    def parse_test_msg(self, msg: Any):
        pieces = []
        for key, disp_key, ty in TEST_MSG_FORMAT:
            value = msg.get(key, 0)
            pieces.append(self._format(disp_key, value, ty).ljust(13, " "))
        return " | ".join(pieces)

    def log(self, level: int, msg: Any):
        if level == INFO:
            self._logger.info(msg)
        elif level == DEBUG:
            self._logger.debug(msg)
        elif level == ERROR:
            self._logger.error(msg)
        elif level == TRAIN:
            self._logger.train(self.parse_train_msg(msg))
            # save data
            self._dump_to_csv(self._train_file, msg, self._train_file_write_header)
            self._train_file_write_header = False
        elif level == TEST:
            self._logger.test(self.parse_test_msg(msg))
            # save data
            self._dump_to_csv(self._test_file, msg, self._test_file_write_header)
            self._test_file_write_header = False
        else:
            raise NotImplementedError

    def _dump_to_csv(self, file: Path, data: Dict, write_header: bool):
        csv_file = file.open("a")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=sorted(data.keys()), restval=0.0
        )

        if write_header:
            csv_writer.writeheader()

        csv_writer.writerow(data)
        csv_file.flush()
