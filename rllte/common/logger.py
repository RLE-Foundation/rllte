# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


import csv
import datetime
from pathlib import Path
from typing import Any, Dict

from termcolor import colored

TRAIN_MSG_FORMAT = [
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("fps", "FPS", "float"),
    ("total_time", "T", "time"),
]

EVAL_MSG_FORMAT = [
    ("step", "S", "int"),
    ("episode", "E", "int"),
    ("episode_length", "L", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]

NUMBER_OF_METRIC_SPACES = 14
NUMBER_OF_PREFIX_SPACES = 5

INFO_PREFIX = "[" + colored("INFO.".ljust(NUMBER_OF_PREFIX_SPACES, " "), "blue", attrs=["bold"]) + "] - "
DEBUG_PREFIX = "[" + colored("DEBUG".ljust(NUMBER_OF_PREFIX_SPACES, " "), "yellow", attrs=["bold"]) + "] - "
ERROR_PREFIX = "[" + colored("ERROR".ljust(NUMBER_OF_PREFIX_SPACES, " "), "white", attrs=["bold"]) + "] - "
TRAIN_PREFIX = "[" + colored("TRAIN".ljust(NUMBER_OF_PREFIX_SPACES, " "), "red", attrs=["bold"]) + "] - "
EVAL_PREFIX = "[" + colored("EVAL.".ljust(NUMBER_OF_PREFIX_SPACES, " "), "green", attrs=["bold"]) + "] - "


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
        self._eval_file = self._log_dir / "eval.log"
        self._train_file_write_header = True
        self._eval_file_write_header = True

    def _format(self, key: str, value: Any, ty: str) -> str:
        """Format the value according to the type.

        Args:
            key (str): The key of the value.
            value (Any): The value to be formatted.
            ty (str): The type of the value.

        Returns:
            The formatted string.
        """
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

    def parse_train_msg(self, msg: Dict) -> str:
        """Parse the training message.

        Args:
            msg (Dict): The training message.

        Returns:
            The formatted string.
        """
        pieces = []
        for key, disp_key, ty in TRAIN_MSG_FORMAT:
            value = msg.get(key, 0)
            pieces.append(self._format(disp_key, value, ty).ljust(NUMBER_OF_METRIC_SPACES, " "))
        return " | ".join(pieces)

    def parse_eval_msg(self, msg: Dict) -> str:
        """Parse the evaluation message.

        Args:
            msg (Dict): The evaluation message.

        Returns:
            The formatted string.
        """
        pieces = []
        for key, disp_key, ty in EVAL_MSG_FORMAT:
            value = msg.get(key, 0)
            pieces.append(self._format(disp_key, value, ty).ljust(NUMBER_OF_METRIC_SPACES, " "))
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
        print(self.time_stamp + INFO_PREFIX + msg)

    def debug(self, msg: str) -> None:
        """Output msg with 'debug' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        print(self.time_stamp + DEBUG_PREFIX + msg)

    def error(self, msg: str) -> None:
        """Output msg with 'error' level.

        Args:
            msg (str): Message to be printed.

        Returns:
            None.
        """
        print(self.time_stamp + ERROR_PREFIX + msg)

    def train(self, msg: Dict) -> None:
        """Output msg with 'train' level.

        Args:
            msg (Dict): Message to be printed.

        Returns:
            None.
        """
        print(self.time_stamp + TRAIN_PREFIX + self.parse_train_msg(msg))
        # save data
        self._dump_to_csv(self._train_file, msg, self._train_file_write_header)
        self._train_file_write_header = False

    def eval(self, msg: Dict) -> None:
        """Output msg with 'eval' level.

        Args:
            msg (Dict): Message to be printed.

        Returns:
            None.
        """
        print(self.time_stamp + EVAL_PREFIX + self.parse_eval_msg(msg))
        # save data
        self._dump_to_csv(self._eval_file, msg, self._eval_file_write_header)
        self._eval_file_write_header = False

    def _dump_to_csv(self, file: Path, data: Dict, write_header: bool) -> None:
        """Dump data to csv file.

        Args:
            file (Path): The file to be written.
            data (Dict): The data to be written.
            write_header (bool): Whether to write the header.

        Returns:
            None.
        """
        csv_file = file.open("a")
        csv_writer = csv.DictWriter(csv_file, fieldnames=sorted(data.keys()), restval=0.0)

        if write_header:
            csv_writer.writeheader()

        csv_writer.writerow(data)
        csv_file.flush()
