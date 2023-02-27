from termcolor import colored

import csv

from hsuanwu.common.typing import *

class Logger:
    """
    The logger class.

    :param log_dir: The logging location.
    """
    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._train_file = self._log_dir / 'train.log'
        self._eval_file = self._log_dir /  'eval.log'
        self._train_file_write_header = True
        self._eval_file_write_header = True

    def log(self, metric, mode='train') -> None:
        if mode == 'train':
            prefix = '[' + colored('Train'.center(7, ' '), 'red', attrs=['bold']) + ']'
        else:
            prefix = '[' + colored('Eval'.center(7, ' '), 'green', attrs=['bold']) + ']'
        
        row = ' | '
        row = row.join([(key + ': ' + metric[key]).center(10) for key in metric.keys()])
        
        print(prefix + ' | ' + row)

        self.dump(metric, mode)


    def dump(self, metric, mode) -> None:
        if mode == 'train':
            self._dump_to_csv(self._train_file, metric, self._train_file_write_header)
            self._train_file_write_header = False
        else:
            self._dump_to_csv(self._eval_file, metric, self._eval_file_write_header)
            self._eval_file_write_header = False


    def _dump_to_csv(self, file, data, write_header):
        csv_file = file.open('a')
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=sorted(data.keys()),
            restval=0.0)
        
        if write_header:
            csv_writer.writeheader()

        csv_writer.writerow(data)
        csv_file.flush()
