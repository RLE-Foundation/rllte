from torch import nn

from hsuanwu.common.typing import *
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init


class MobileNetV1Encoder(BaseEncoder):
    """
    MobileNetV1 encoder for processing image-based observations.
    Currently this version only contains one fixed block.

    Args:
        observation_space: Observation space of the environment.
        feature_dim: Number of features extracted.
    Returns:
        MobileNetV1 encoder instance.
    """
    def __init__(self, observation_space: Space, feature_dim: int = 64) -> None:
        super().__init__(observation_space, feature_dim)
        obs_shape = observation_space.shape
        if len(obs_shape) == 4:
            # vectorized envs
            obs_shape = obs_shape[1:]
        assert len(obs_shape) == 3

        def conv_bn(inp, oup, stride) -> torch.nn.modules.container.Sequential:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )
        
        def conv_dw(inp, oup, stride) -> torch.nn.modules.container.Sequential:
            return nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # Pairwise convolution
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )
        
        self.trunk = nn.Sequential(
            conv_bn(obs_shape[0], 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    
        self.linear = nn.Linear(1024, feature_dim)
        self.apply(network_init)

    def forward(self, obs: Tensor) -> Tensor:
        h = self.trunk(obs)
        return self.linear(h)