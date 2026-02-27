"""
Layer Normalization
https://arxiv.org/pdf/1607.06450
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.eps = 1e-6

        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: torch.Tensor): # (..., dim)

        # collect statistics of individual feature vector
        mean = x.mean(dim=-1, keepdim=True) # (..., 1)
        var = x.var(dim=-1, keepdim=True, correction=0) # (..., 1)
        stdev = (var + self.eps) ** 0.5 # (..., 1)

        # normalize input using collected statistics
        normalized = (x - mean) / stdev  # (..., dim)

        # scale normalized input
        scaled = normalized * self.gamma + self.beta  # (..., dim)

        return scaled