import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch.multiprocessing as mp


class Test:
    def __init__(self) -> None:
        ctx = mp.get_context("fork")
        self.args1 = 1
        self.args2 = 2
        self.args3 = 3

        actor_pool = list()
        for i in range(5):
            actor = ctx.Process(target=self.act, args=(self.args1, self.args2))
            actor.start()
            actor_pool.append(actor)

    def act(self, arg1, arg2):
        print("Working!")
        print(arg1 + arg2 + self.args3)


if __name__ == "__main__":
    test = Test()
