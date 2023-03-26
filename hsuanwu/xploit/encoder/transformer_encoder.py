from torch import nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from hsuanwu.common.typing import *
from hsuanwu.xploit.encoder.base import BaseEncoder
from hsuanwu.xploit.utils import network_init


class PatchEmbedding(nn.Module):
    """Patch embedding for transformers with image inputs

    Args:
        in_channels: Channels of inputs.
        patch_size:  Channels of outputs
        emb_size: size of embedding for each patch
        img_size: height and width of the image
    """
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 32) -> None:
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    

class LinearHead(nn.Sequential):
    """linear layer for outputting the feature

    Args:
        emb_size: input of the linear layer
        feature_dim: output of the linear layer
    """
    def __init__(self, emb_size: int , feature_dim: int) -> None:
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, feature_dim))
        


class TransformerEcoder(BaseEncoder):
    """
    Transformer encoder for processing image-based observations. The

    Args:
        observation_space: Observation space of the environment.
        feature_dim: Number of features extracted.
    Returns:
        Transformer encoder instance.
    """
    def __init__(self, observation_space: Space, feature_dim: int = 64, patch_size: int = 4, \
                 emb_size: int = 768, nhead: int = 8, dim_feedforward: int = 2048, num_layers: int = 6) -> None:
        super().__init__(observation_space, feature_dim)
        obs_shape = observation_space.shape
        if len(obs_shape) == 4:
            # vectorized envs
            obs_shape = obs_shape[1:]
        assert len(obs_shape) == 3
        assert obs_shape[1] == obs_shape[2]
        self.embedding = PatchEmbedding(in_channels=obs_shape[0], patch_size=patch_size, emb_size=emb_size, img_size = obs_shape[1])
        self.encoder = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.trunk = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)
        self.linear = LinearHead(emb_size=emb_size, feature_dim=feature_dim)

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs / 255.
        h = self.embedding(obs)
        h = self.trunk(h)
        return self.linear(h)
        

