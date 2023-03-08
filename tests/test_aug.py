import os
import sys

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

from hsuanwu.xplore.augmentation import RandomCrop
from hsuanwu.env.dmc import make_dmc_env

import torch
import numpy as np
import cv2

# from torch import nn
# import kornia
# class Crop(object):
#     """
#     Crop Augmentation
#     """
#     def __init__(self,  
#                  batch_size, 
#                  *_args, 
#                  **_kwargs):
#         self.batch_size = batch_size 

#     def do_augmentation(self, x):
#         aug_trans = nn.Sequential(nn.ReplicationPad2d(12),
#                             kornia.augmentation.RandomCrop((84, 84)))
#         return aug_trans(x)

#     def change_randomization_params(self, index_):
#         pass

#     def change_randomization_params_all(self):
#         pass

#     def print_parms(self):
#         pass

if __name__ == '__main__':
    env = make_dmc_env(domain_name='hopper', 
                       task_name='hop', 
                       resource_files=None, 
                       img_source=None,
                       total_frames=None,
                       seed=1, 
                       visualize_reward=False, 
                       from_pixels=True, 
                       frame_skip=1)
    
    obs = env.reset()
    print(obs.shape)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0) / 255.0
    cv2.imwrite('./tests/origin.png', np.transpose(obs_tensor.numpy()[0], [1, 2, 0]) * 255.0)
    print(obs_tensor.size())

    aug = RandomCrop(pad=20, out=84)

    # auged_obs = Crop(1).do_augmentation(obs_tensor) 
    auged_obs = aug(obs_tensor)
    print(auged_obs.size())
    cv2.imwrite('./tests/after.png', np.transpose(auged_obs.numpy()[0], [1, 2, 0]) * 255.0)