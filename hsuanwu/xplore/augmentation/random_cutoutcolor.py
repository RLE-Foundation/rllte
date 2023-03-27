import torch
from hsuanwu.common.typing import *
from hsuanwu.xplore.augmentation.base import BaseAugmentation

class RandomCutoutColor(BaseAugmentation):

    """ Random Cutout operation for image augmentation.

    Args: the size of the cut area 
        min_cut: min size of the cut shape.
        max_cut: max size of the cut shape.
    
    Returns:
       Random Cutout Colored image.

    """

    def __init__(self, min_cut=10, max_cut=30)->None:
        super().__init__()
        self.min_cut = min_cut
        self.max_cut = max_cut
        
    def forward(self, imgs):
        n, c, h, w = imgs.shape
        
        w1 = torch.randint(self.min_cut, self.max_cut, (n,))
        h1 = torch.randint(self.min_cut, self.max_cut, (n,))
        
        cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
        rand_box = torch.rand((n, c), device=imgs.device)
        
        for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
            cut_img = img.clone()
            
            rand_color = rand_box[i].reshape(-1, 1, 1).expand_as(cut_img[:, h11:h11+h11, w11:w11+w11])
            cut_img[:, h11:h11+h11, w11:w11+w11] = rand_color
            
            cutouts[i] = cut_img
            
        return cutouts