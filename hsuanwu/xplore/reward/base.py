from hsuanwu.common.typing import *

class BaseIntrinsicRewardModule(object):
    """Base class of intrinsic reward module.

    Args:
        env: The environment.
        device: Device (cpu, cuda, ...) on which the code should be run.
        beta: The initial weighting coefficient of the intrinsic rewards.
        kappa: The decay rate.
    
    Returns:
        Instance of the base intrinsic reward module.
    """
    def __init__(
            self,
            env: Env,
            device: torch.device,
            beta: float,
            kappa: float
            ) -> None:
        
        self._obs_shape = env.observation_space.shape
        if env.action_space.__class__.__name__ == 'Discrete':
            self._action_shape = env.action_space.n
            self._action_type = 'dis'
        elif env.action_space.__class__.__name__ == 'Box':
            self._action_shape = env.action_space.shape
            self._action_type = 'cont'
        else:
            raise NotImplementedError

        self._device = device
        self._beta = beta
        self._kappa = kappa

    def compute_irs(self, rollouts: Dict, step: int) -> ndarray:
        """Compute the intrinsic rewards using the collected observations.

        Args:
            rollouts: The collected experiences. A python dict like 
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
            step: The current time step.
        
        Returns: 
            The intrinsic rewards.
        """
        pass

    def update(self, rollouts: Dict,) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            rollouts: The collected experiences. A python dict like 
                {observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>,
                actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>,
                rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>}.
        
        Returns:
            None
        """
        pass