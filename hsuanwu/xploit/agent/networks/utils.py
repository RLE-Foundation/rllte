from typing import Callable
from torch import nn

def get_network_init(method: str = "orthogonal") -> Callable:
    """Returns a network initialization function.

    Args:
        method (str): Initialization method name.

    Returns:
        Initialization function.
    """
    def _identity(m):
        """Identity initialization."""
        pass

    def _orthogonal(m):
        """Orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(m.weight.data, gain)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
    
    def _xavier_uniform(m):
        """Xavier uniform initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
    
    def _xavier_normal(m):
        """Xavier normal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    if method == "orthogonal":
        return _orthogonal
    elif method == "xavier_normal":
        return _xavier_normal
    elif method == "xavier_uniform":
        return _xavier_uniform
    else:
        return _identity