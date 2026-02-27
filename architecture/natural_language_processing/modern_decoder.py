"""
LLaMA: Open and Efficient Foundation Language Models
https://arxiv.org/pdf/2302.13971
"""

import torch
from torch import nn
from multiheadattention import MultiheadAttention
from normalization.rmsnorm import RMSNorm
from mlp import MLP
from function import silu

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_ratio):
        super().__init__()

        self.attention = MultiheadAttention(embed_dim, heads, causal=True) # causal self attention
        self.mlp = MLP(embed_dim, bias=False, activation=silu, mlp_ratio=mlp_ratio, use_glu=True) # swish gated linear unit

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x): # (batch, token, embed_dim)

        # pre norm outside the residual stream improving training stability
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x # (batch, token, embed_dim)