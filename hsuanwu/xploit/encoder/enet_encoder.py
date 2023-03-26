import torch
from torch import nn

from hsuanwu.common.typing import *
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init


class InitialBlock(nn.Module):
    """Initial block for Enet taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py

    Args:
        in_channels: Channels of inputs.
        out_channels： Channels of outputs
        bias: Adding a learnable bias to the output, set to False by default
        relu: using ReLU for activation if True and PReLU for activation if False
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, relu: bool = True) -> None:
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=bias)
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """Regular block for Enet taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py

    Args:
        channels: the number of input and output channels.
        internal_ratio: a scale factor applied to ``channels`` used to compute the number of channels after the projection. 
        kernel_size: the kernel size of the filters used in the convolution layer described above in item 2 of the extension branch..
        padding: zero-padding added to both sides of the input.
        dilation (int, optional): spacing between kernel elements for the convolution described in item 2 of the extension branch.
        asymmetric: flags if the convolution described in item 2 of the extension branch is asymmetric or not.
        dropout_prob (float, optional): probability of an element to be zeroed.
        bias (bool, optional): Adds a learnable bias to the output if ``True``.
        relu (bool, optional): When ``True`` ReLU is used as the activation function; otherwise, PReLU is used.
    """
    def __init__(self, channels: int,
                internal_ratio: int = 4, 
                kernel_size: int = 3, 
                padding: int = 0, 
                dilation: int = 1, 
                asymmetric: bool = False, 
                dropout_prob: float = 0, 
                bias: bool = False, 
                relu: bool = True) -> None:
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU


        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias), 
            nn.BatchNorm2d(internal_channels), 
            activation())

        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=bias),
                nn.BatchNorm2d(internal_channels), 
                activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=1,  padding=(0, padding), dilation=dilation, bias=bias), 
                nn.BatchNorm2d(internal_channels), 
                activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias), 
                nn.BatchNorm2d(internal_channels), 
                activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias), 
            nn.BatchNorm2d(channels), 
            activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        # Main branch shortcut
        main = x
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    """Down sampling block for Enet taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/models/enet.py

    Args:
        in_channels: the number of input channels.
        out_channels: the number of output channels.
        internal_ratio: a scale factor applied to ``channels`` used to compute the number of channels after the projection.
        return_indices:  if ``True``, will return the max indices along with the outputs. Useful when unpooling later.
        dropout_prob: probability of an element to be zeroed.
        bias: Adds a learnable bias to the output if ``True``.
        relu (bool, optional): When ``True`` ReLU is used as the activation function; otherwise, PReLU is used.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 internal_ratio: int = 4,
                 return_indices: bool = False,
                 dropout_prob: float = 0,
                 bias: bool = False,
                 relu=True) -> None:
        super().__init__()


        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias), 
            nn.BatchNorm2d(internal_channels), 
            activation())

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.BatchNorm2d(internal_channels), 
            activation())

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias), 
            nn.BatchNorm2d(out_channels), 
            activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)




class ENetEncoder(BaseEncoder):
    """
    Encoder part of ENet for processing image-based observations.
    The encoder consists of 3 stages, the nummber of input channels for each stage should be specified

    Args:
        observation_space: Observation space of the environment.
        stage1_input_dim: nummber of channels for stage 1's block
        stage2_input_dim: nummber of channels for stage 2's block
        stage3_input_dim: nummber of channels for stage 3's block
        feature_dim: Number of features extracted.
        encoder_relu: When ``True`` ReLU is used as the activation function in the encoder blocks/layers; otherwise, PReLU is used.

    Returns:
        ENet-like encoder instance.
    """


    def __init__(self, observation_space: Space,
                feature_dim: int = 0, 
                stage1_input_dim: int = 0,
                stage2_input_dim: int = 0,
                stage3_input_dim: int = 0,
                encoder_relu=False):
        super().__init__(observation_space, feature_dim)
        modules = list()
        obs_shape = observation_space.shape
        if len(obs_shape) == 4:
            # vectorized envs
            obs_shape = obs_shape[1:]
        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2]
        assert obs_shape[1] % 8 == 0 # ENet has 3 stages of encoders, the height and width after the encoders would be divided by 8
        
        modules.append(InitialBlock(obs_shape[0], stage1_input_dim, relu=encoder_relu))

        # Stage 1 - Encoder
        modules.append(DownsamplingBottleneck(stage1_input_dim, stage2_input_dim, return_indices=True, dropout_prob=0.01, relu=encoder_relu)) # downsample1_0
        modules.append(RegularBottleneck(stage2_input_dim, padding=1, dropout_prob=0.01, relu=encoder_relu)) # regular1_1
        modules.append(RegularBottleneck(stage2_input_dim, padding=1, dropout_prob=0.01, relu=encoder_relu)) # regular1_2
        modules.append(RegularBottleneck(stage2_input_dim, padding=1, dropout_prob=0.01, relu=encoder_relu)) # regular1_3
        modules.append(RegularBottleneck(stage2_input_dim, padding=1, dropout_prob=0.01, relu=encoder_relu)) # regular1_4

        # Stage 2 - Encoder
        modules.append(DownsamplingBottleneck(stage2_input_dim, stage3_input_dim, return_indices=True, dropout_prob=0.1, relu=encoder_relu)) # downsample2_0
        modules.append(RegularBottleneck(stage3_input_dim, padding=1, dropout_prob=0.1, relu=encoder_relu)) # regular2_1
        modules.append(RegularBottleneck(stage3_input_dim, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)) # regular2_2
        modules.append(RegularBottleneck(stage3_input_dim, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)) # asymmetric2_3
        modules.append(RegularBottleneck(stage3_input_dim, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)) # dilated2_4
        modules.append(RegularBottleneck(stage3_input_dim, padding=1, dropout_prob=0.1, relu=encoder_relu)) # regular2_5
        modules.append(RegularBottleneck(stage3_input_dim, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)) # dilated2_6
        modules.append(RegularBottleneck(stage3_input_dim, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)) # asymmetric2_7
        modules.append(RegularBottleneck(stage3_input_dim, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)) # dilated2_8

        # Stage 3 - Encoder
        modules.append(RegularBottleneck(stage3_input_dim, padding=1, dropout_prob=0.1, relu=encoder_relu)) # regular3_0
        modules.append(RegularBottleneck(stage3_input_dim, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)) # dilated3_1
        modules.append(RegularBottleneck(stage3_input_dim, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)) # asymmetric3_2
        modules.append(RegularBottleneck(stage3_input_dim, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)) # dilated3_3
        modules.append(RegularBottleneck(stage3_input_dim, padding=1, dropout_prob=0.1, relu=encoder_relu)) # regular3_4
        modules.append(RegularBottleneck(stage3_input_dim, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)) # dilated3_5
        modules.append(RegularBottleneck(stage3_input_dim, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)) # asymmetric3_6
        modules.append(RegularBottleneck(stage3_input_dim, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)) # dilated3_7

        modules.append(nn.Flatten())
        self.trunk = nn.Sequential(*modules)

        with torch.no_grad():
            sample = torch.ones(size=tuple(obs_shape)).float()
            n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

        self.linear = nn.Linear(n_flatten, out_features=feature_dim)

        self.apply(network_init)

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs / 255.
        h = self.trunk(obs)
        return self.linear(h)