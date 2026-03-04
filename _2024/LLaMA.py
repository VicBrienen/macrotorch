"""
The Llama 3 Herd of Models
https://arxiv.org/pdf/2407.21783
"""

import torch
from torch import nn
from _2017.silu import silu
from _2019.rmsnorm import RMSNorm
from _2020.glu import GLU
from _2021.rope import RoPE
from _2023.gqa import GroupedQueryAttention

# very similar to most open source LLMs

class LLaMABlock(nn.Module):
    def __init__(self, embed_dim, heads, groups):
        super().__init__()

        head_dim = embed_dim // heads
        rope = RoPE(head_dim)

        self.attention = GroupedQueryAttention(embed_dim, heads, groups, rope, causal=True) # causal self attention
        self.mlp = GLU(embed_dim, bias=False, activation=silu) # swish gated linear unit

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x): # (batch, token, embed_dim)

        # pre norm outside the residual stream improving training stability
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x # (batch, token, embed_dim)