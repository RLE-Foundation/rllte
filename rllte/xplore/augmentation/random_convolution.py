import torch as th

from rllte.common.base_augmentation import BaseAugmentation


class RandomConvolution(BaseAugmentation):
    """Random Convolution operation for image augmentation. Note that imgs should be normalized and torch tensor.

    Args:
        None.

    Returns:
        Augmented images.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        num_batch, num_stack_channel, img_h, img_w = x.size()
        num_trans = num_batch
        batch_size = int(num_batch / num_trans)

        rand_conv = th.nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(x.device)

        for trans_index in range(num_trans):
            th.nn.init.xavier_normal_(self.rand_conv.weight.data)
            temp_imgs = x[trans_index * batch_size : (trans_index + 1) * batch_size]
            temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w)  # (batch x stack, channel, h, w)
            rand_out = rand_conv(temp_imgs)
            if trans_index == 0:
                total_out = rand_out
            else:
                total_out = th.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)

        return total_out
