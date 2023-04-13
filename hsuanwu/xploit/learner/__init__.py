from .base import BaseLearner
from .drqv2 import DEFAULT_CFGS as DRQV2_DEFAULT_CFGS
from .drqv2 import DrQv2Learner
from .impala import DEFAULT_CFGS as IMPALA_DEFAULT_CFGS
from .impala import IMPALALearner
from .network import *
from .ppg import DEFAULT_CFGS as PPG_DEFAULT_CFGS
from .ppg import PPGLearner
from .ppo import DEFAULT_CFGS as PPO_DEFAULT_CFGS
from .ppo import PPOLearner
from .sac import DEFAULT_CFGS as SAC_DEFAULT_CFGS
from .sac import SACLearner

ALL_DEFAULT_CFGS = {
    "DrQv2Learner": DRQV2_DEFAULT_CFGS,
    "PPGLearner": PPG_DEFAULT_CFGS,
    "SACLearner": SAC_DEFAULT_CFGS,
    "PPOLearner": PPO_DEFAULT_CFGS,
    "IMPALALearner": IMPALA_DEFAULT_CFGS,
}
