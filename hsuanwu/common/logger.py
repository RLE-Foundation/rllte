import csv
import datetime
from pathlib import Path
from typing import Any, Dict, Type

from termcolor import colored

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

        self._train_file = self._log_dir / "train.log"
        self._test_file = self._log_dir / "test.log"
        self._train_file_write_header = True
        self._test_file_write_header = True

    def _format(self, key: str, value: Any, ty: str):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.03f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{key}: {value}"
        else:
            raise TypeError(f"invalid format type: {ty}")

    def parse_train_msg(self, msg: Any) -> str:
        pieces = []
        for key, disp_key, ty in TRAIN_MSG_FORMAT:
            value = msg.get(key, 0)
            pieces.append(self._format(disp_key, value, ty).ljust(14, " "))
        return " | ".join(pieces)

    def parse_test_msg(self, msg: Any) -> str:
        pieces = []
        for key, disp_key, ty in TEST_MSG_FORMAT:
            value = msg.get(key, 0)
            pieces.append(self._format(disp_key, value, ty).ljust(14, " "))
        return " | ".join(pieces)

    @property
    def time_stamp(self) -> str:
        """Return the current time stamp."""
        return "[" + datetime.datetime.now().strftime("%m/%d/%Y %I:%M:%S %p") + "] - "

    def info(self, msg: str) -> None:
        """Output msg with 'info' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        prefix = "[" + colored("HSUANWU INFO".ljust(13, " "), "cyan", attrs=["bold"]) + "] - "
        print(self.time_stamp + prefix + msg)

    def debug(self, msg: str) -> None:
        """Output msg with 'debug' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        prefix = "[" + colored("HSUANWU DEBUG".ljust(13, " "), "yellow", attrs=["bold"]) + "] - "
        print(self.time_stamp + prefix + msg)

    def error(self, msg: str) -> None:
        """Output msg with 'error' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        prefix = "[" + colored("HSUANWU ERROR".ljust(13, " "), "white", attrs=["bold"]) + "] - "
        print(self.time_stamp + prefix + msg)

    def train(self, msg: Dict) -> None:
        """Output msg with 'train' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        prefix = "[" + colored("HSUANWU TRAIN".ljust(13, " "), "red", attrs=["bold"]) + "] - "
        print(self.time_stamp + prefix + self.parse_train_msg(msg))
        # save data
        self._dump_to_csv(self._train_file, msg, self._train_file_write_header)
        self._train_file_write_header = False

    def test(self, msg: Dict) -> None:
        """Output msg with 'test' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        prefix = "[" + colored("HSUANWU TEST".ljust(13, " "), "green", attrs=["bold"]) + "] - "
        print(self.time_stamp + prefix + self.parse_test_msg(msg))
        # save data
        self._dump_to_csv(self._test_file, msg, self._test_file_write_header)
        self._test_file_write_header = False

    def _dump_to_csv(self, file: Path, data: Dict, write_header: bool) -> None:
        csv_file = file.open("a")
        csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(data.keys()), restval=0.0)

        if write_header:
            csv_writer.writeheader()

        csv_writer.writerow(data)
        csv_file.flush()
