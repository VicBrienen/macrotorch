"""
Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385
"""

import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, layer, shortcut=None):
        super().__init__()

        # the actual layer type (MLP, convolutional, attention, etc.)
        self.layer = layer

        # if dimensions match use identity mapping, otherwise a shortcut function needs to be provided (usually a linear projection)
        self.shortcut = shortcut if shortcut is not None else nn.Identity()

    def forward(self, x):
        
        x = self.shortcut(x) + self.layer(x) # residual

        return x