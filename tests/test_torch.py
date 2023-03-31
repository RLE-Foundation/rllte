import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from torch import nn
import torch
import time

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
    
    def forward(self, obs):
        return self.main(obs)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    encoder = Encoder().to(device)

    t_s = time.perf_counter()
    for i in range(10):
        encoder(torch.rand(128, 3, 84, 84, device=device))
    t_e = time.perf_counter()
    print('Uncompiled: ', t_e - t_s)

    t_s = time.perf_counter()
    encoder = torch.compile(encoder)
    for i in range(10):
        encoder(torch.rand(128, 3, 84, 84, device=device))
    t_e = time.perf_counter()
    print('Compiled: ', t_e - t_s)