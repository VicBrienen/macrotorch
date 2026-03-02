"""
The perceptron: a probabilistic model for information storage and organization in the brain.
https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf
"""

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.01)

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