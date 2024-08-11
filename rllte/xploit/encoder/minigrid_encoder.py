


import torch
from torch import nn
from rllte.common.prototype import BaseEncoder

class MinigridEncoder(BaseEncoder):
    def __init__(self, observation_space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            observations = observation_space.sample()
            observations = torch.as_tensor(observations[None]).float()
            n_flatten = self.cnn(observations).float().shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #observations = observations.permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(observations.float()))