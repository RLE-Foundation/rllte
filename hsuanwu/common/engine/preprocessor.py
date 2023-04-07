from omegaconf import OmegaConf, open_dict
from hsuanwu.common.typing import DictConfig
from hsuanwu.common.logger import Logger, DEBUG
from hsuanwu.xploit.learner import ALL_DEFAULT_CFGS


def cfgs_preprocessor(logger: Logger, cfgs: DictConfig):
    """Preprocess the configs.
    
    Args:
        cfgs (DictConfig): configs.
    
    Returns:
        Processed configs.
    """
    new_cfgs = OmegaConf.create()




    return new_cfgs