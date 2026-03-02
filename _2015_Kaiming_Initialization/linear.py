"""
Kaiming initialization
https://arxiv.org/pdf/1502.01852
"""

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        
        self.weights = nn.Parameter(torch.randn((in_features, out_features)) * (2 / in_features) ** 0.5)

        # optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x): # (batch, in_features)
        x = x @ self.weights

        if self.bias is not None:
            x = x + self.bias
        
        return x # (batch, out_features)