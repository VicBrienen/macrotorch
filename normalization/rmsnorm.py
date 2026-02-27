"""
https://arxiv.org/pdf/1910.07467 - Root Mean Square Layer Normalization
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.eps = 1e-6 # prevent division by 0 and ensure gradient stability
        self.gain = nn.Parameter(torch.ones(dim)) # learnable per feature scaling parameter

    def forward(self, x): # (..., dim)

        # root mean square operation on the last dimension on the input tensor
        square = x ** 2 # (..., dim)
        mean = torch.mean(square, dim=-1, keepdim=True) # (..., 1)
        rms = torch.sqrt(mean + self.eps) # (..., 1)

        # divide features by rms and multiply with learned gains
        rmsnorm =  (x / rms) * self.gain # (..., dim)

        return rmsnorm