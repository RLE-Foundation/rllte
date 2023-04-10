from .base import BaseLearner
from .drac import DrACLearner
from .drqv2 import DrQv2Learner
from .impala import IMPALALearner
from .network import *
from .ppg import PPGLearner
from .ppo import PPOLearner
from .sac import SACLearner

from .drqv2 import DEFAULT_CFGS as DRQV2_DEFAULT_CFGS
from .sac import DEFAULT_CFGS as SAC_DEFAULT_CFGS

ALL_DEFAULT_CFGS = {
    'DrQv2Learner': DRQV2_DEFAULT_CFGS,
    'SACLearner': SAC_DEFAULT_CFGS
}