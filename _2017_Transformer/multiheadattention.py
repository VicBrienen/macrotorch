"""
Attention Is All You Need
https://arxiv.org/pdf/1706.03762
"""

import torch
from torch import nn
from _2015_Residual_Connection.kaiming import Linear


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, heads, causal=False):
        super().__init__()

        # convention of dividing the embed_dim by number of heads to obtain head_dim for computational purposes
        self.heads = heads
        self.head_dim = embed_dim // heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)

        self.out_proj = Linear(embed_dim, embed_dim)

        self.causal = causal

    def forward(self, x): # (batch, token, embed_dim)
        batch, token, embed_dim = x.shape # extract input shape

        # linear query, key, and value projections (batch, token, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split q, k, and v into multiple heads (batch, token, heads, head_dim)
        q = q.reshape(batch, token, self.heads, self.head_dim)
        k = k.reshape(batch, token, self.heads, self.head_dim)
        v = v.reshape(batch, token, self.heads, self.head_dim)

        # reorder q, k, and v dimensions (batch, heads, token, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # for each head, perform self attention between queries and keys
        attention_scores = q @ k.permute(0, 1, 3, 2) # (batch, heads, token, token)

        # variance of dot product is head_dim. to make variance 1 we divide by the standard deviation head_dim ** 0.5
        # so that input to softmax does not result in near binary output causing vanishing gradients
        attention_scores = attention_scores / self.head_dim ** 0.5 # (batch, heads, token, token)

        if self.causal: # add a causal mask to prevent infromation leakage for autoregressive training
            mask = torch.triu(torch.ones(token, token, device=x.device, dtype=torch.bool), diagonal=1) # (token, token)
            attention_scores = attention_scores.masked_fill(mask, float("-inf")) # (batch, heads, token, token)


        # apply softmax to turn scores into probabilities that sum to 1 and turn attention into weighted average
        attention_weights = torch.softmax(attention_scores, dim=-1) # (batch, heads, token, token)

        # dot product between attention weights and corresponding value vectors
        out = attention_weights @ v # (batch, heads, token, head_dim)

        # rorder dimensions and concatenate heads 
        out = out.permute(0, 2, 1, 3) # (batch, token, heads, head_dim)
        out = out.reshape(batch, token, embed_dim) # (batch, token, embed_dim)

        # linear output projection to mix head information
        out = self.out_proj(out) # (batch, token, embed_dim)

        return out