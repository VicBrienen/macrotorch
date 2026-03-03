"""
GLU Variants Improve Transformer
https://arxiv.org/pdf/2002.05202
"""

import torch
from torch import nn
from _2015.kaiming import Linear
from _2016.glu import glu

class GLU(nn.Module):
    def __init__(self, dim, bias=False, activation=None, mlp_ratio=4, use_glu=True):
        super().__init__()

        self.activation = activation
        self.use_glu = use_glu

        if self.use_glu:
            mlp_ratio = 4 * 2 / 3 # ensure same parameter count as regular MLP
            hidden_dim = int(dim * mlp_ratio)
            self.linear_gate = Linear(dim, hidden_dim, bias) # additional upward gating projection for glu

        else:
            hidden_dim = dim * mlp_ratio

        # regular MLP up and down projection
        self.linear1 = Linear(dim, hidden_dim, bias)
        self.linear2 = Linear(hidden_dim, dim, bias)

    def forward(self, x):

        if self.use_glu:
            gate_value = self.linear_gate(x)
            x = self.linear1(x)
            x = glu(x, gate_value, self.activation)

        else:
            x = self.linear1(x)
            x = self.activation(x)

        x = self.linear2(x) # standard down projection back to input dimension

        return x