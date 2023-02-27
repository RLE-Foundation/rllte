from hsuanwu.common.typing import *

class BaseRewardModule(object):
    """
    Base class of intrinsic reward module.
    
    :param env: The environment.
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    :param beta: The initial weighting coefficient of the intrinsic rewards.
    :param kappa: The decay rate.
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
        """
        Compute the intrinsic rewards using the collected observations.

        :param rollouts: The collected experiences. A python dict like:
            + observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>
            - actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>
            + rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>
        :param step: The current time step.
        
        :return: The intrinsic rewards
        """
        pass

    def update(self, rollouts: Dict,) -> None:
        """
        Update the intrinsic reward module if necessary.

        :param rollouts: The collected experiences. A python dict like:
            + observations (n_steps, n_envs, *obs_shape) <class 'numpy.ndarray'>
            - actions (n_steps, n_envs, action_shape) <class 'numpy.ndarray'>
            + rewards (n_steps, n_envs, 1) <class 'numpy.ndarray'>
        """
        pass