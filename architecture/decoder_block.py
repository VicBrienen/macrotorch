import torch
from torch import nn
from multiheadattention import MultiheadAttention
from normalization.rmsnorm import RMSNorm
from mlp import MLP

class Block(nn.Module):
    def __init__(self, embed_dim, heads, mlp_ratio):
        super().__init__()

        self.attention = MultiheadAttention(embed_dim, heads)
        self.mlp = MLP(embed_dim, mlp_ratio)

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

    def forward(self, x): # (batch, token, embed_dim)

        # pre layer normalization (prenorm) outside the residual stream for training stability
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x # (batch, token, embed_dim)
    
