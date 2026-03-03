"""
Improving neural networks by preventing co-adaptation of feature detectors
https://arxiv.org/pdf/1207.0580

Dropout: A Simple Way to Prevent Neural Networks from Overfitting
https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b4
"""

import torch
from torch import nn

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p_drop = p # probability of dropping an activation
        self.p_keep = 1 - p # probability of keeping an activation

    def forward(self, x):

        # only apply dropout during training
        if not self.training:
            return x

        shape = x.shape # get shape of x

        # generate binary mask with same shape as x
        mask = (torch.rand(shape, device=x.device) > self.p_drop).to(x.dtype) # ensure on same device and same dtype

        # drop activations based on binary mask
        x = x * mask

        # rescale activations to maintain expected value
        x = x / self.p_keep

        return x