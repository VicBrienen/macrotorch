"""
Improving Language Understanding by Generative Pre-Training
https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
"""

import torch
from torch import nn
from _2017.multiheadattention import MultiheadAttention
from _2016.layernorm import LayerNorm
from _2016.gelu import gelu
from glu import MLP


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_ratio):
        super().__init__()

        self.attention = MultiheadAttention(embed_dim, heads, causal=True) # causal self attention
        self.mlp = MLP(embed_dim, bias=True, activation=gelu, mlp_ratio=mlp_ratio) # regular GELU MLP

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x): # (batch, token, embed_dim)

        # post norm over residual stream in GPT-1 style (not as stable as prenorm)
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))

        return x # (batch, embed_dim, token)