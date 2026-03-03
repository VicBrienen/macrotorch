"""
Gradient-based learning applied to document recognition
http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""

import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(in_features, out_features) / in_features ** 0.5) # scaling init weights using in_features
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x): # (batch, in_features)
        
        x = x @ self.weights + self.bias
        
        return x # (batch, out_features)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.projection = Linear(in_channels * kernel_size ** 2, out_channels)

    def forward(self, x): # (batch, channel, height, width)
        batch, _, height, width = x.shape
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        
        # extract patches (batch, out_height, out_width, in_channels, kernel_size, kernel_size)
        patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # flatten patches (batch, out_height, out_width, in_channels * kernel_size ** 2)
        patches = patches.contiguous().view(batch, out_height, out_width, -1)

        out = self.projection(patches) # (batch, out_height, out_width, out_channels)

        # permute to standard format
        out = out.permute(0, 3, 1, 2) # (batch, out_channels, out_height, out_width)

        return out

        
