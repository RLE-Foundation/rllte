import torch

from hsuanwu.common.typing import *

class BaseLearner:
    """Base class of learner.
    
    Args:
        observation_space: Observation space of the environment.
        action_space: Action space of the environment.
        action_type: Continuous or discrete action. "cont" or "dis".
        device: Device (cpu, cuda, ...) on which the code should be run.
        feature_dim: Number of features extracted.
        lr: The learning rate.
        eps: Term added to the denominator to improve numerical stability.

    Returns:
        Base learner instance.
    """
    def __init__(self,
                observation_space: Space, 
                action_space: Space,
                action_type: str,
                device: torch.device = 'cuda',
                feature_dim: int = 50,
                lr: float = 2.5e-4,
                eps: float = 1e-5
                ) -> None:
        self._obs_space = observation_space
        self._action_space = action_space
        self._action_type = action_type
        self._device = torch.device(device)
        self._feature_dim = feature_dim
        self._lr = lr
        self._eps = eps

        # placeholder for distribution, augmentation, and intrinsic reward function.
        self._dist = None
        self._aug = None
        self._irs = None


    def train(self, training: bool = True) -> None:
        """ Set the train mode.

        Args:
            training: True (training) or False (testing).

        Returns:
            None.
        """
        self.training = training


    def set_encoder(self, encoder: torch.nn.Module) -> None:
        """Set the encoder for learner.
        
        Args:
            encoder: Hsuanwu encoder class.
        
        Returns:
            None.
        """
        self._encoder = encoder
        self._encoder.train()
        self._encoder_opt = torch.optim.Adam(self._encoder.parameters(), lr=self._lr, eps=self._eps)


    def set_dist(self, dist: Distribution) -> None:
        """Set the distribution for earner.
        
        Args:
            dist: Hsuanwu distribution class.
        
        Returns:
            None.
        """
        self._dist = dist


    def set_aug(self, aug) -> None:
        """Set the augmentation for learner.
        
        Args:
            irs: Hsuanwu augmentation class.
        
        Returns:
            None.
        """
        self._aug = aug


    def set_irs(self, irs) -> None:
        """Set the intrinsic reward function for learner.
        
        Args:
            irs: Hsuanwu intrinsic reward class.
        
        Returns:
            None.
        """
        self._irs = irs


    def act(self, obs: Tensor, training: bool = True, *args) -> Tensor:
        """Sample actions.
        
        Args:
            obs: Observation tensor.
            training: True (training) or False (testing).
        
        Returns:
            Sampled actions.
        """
        pass


    def update(self, *args) -> Dict:
        """Update learner.
        
        Args:
            
        
        Returns:
            Training metrics such as loss functions.
        """
        pass