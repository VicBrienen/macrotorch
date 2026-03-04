"""
RoFormer: Enhanced Transformer with Rotary Position Embedding
https://arxiv.org/pdf/2104.09864
"""

import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, dim, max_embeddings=2048, base=10000.0):
        super().__init__()

        self.dim = dim # head dimension
        self.max_embeddings = max_embeddings # maximum expected sequence length
        self.base = base # base value controlling wavelength of rotational frequency

        # angular velocity of the rotation in the complex place of shape (dim / 2)
        inverse_frequencies = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim)) 
        self.register_buffer("inverse_frequencies", inverse_frequencies)

        # generate 1D tensor of positional indices (max_embeddings)
        t = torch.arange(max_embeddings, dtype=inverse_frequencies.dtype, device=inverse_frequencies.device)

        # multiply each position index with each angular velocity resulting in shape (max_embeddings, dim / 2)
        frequencies = torch.outer(t, self.inverse_frequencies)

        # concatenate frequencies along the last dimension to match head dim (max_embeddings, D)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", embedding.cos(), persistent=False)
        self.register_buffer("sin_cached", embedding.sin(), persistent=False)
    
    def _rotate_half(self, x):

        # extract 2 sets of features for each token (batch, heads, seq_len, dim / 2)
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1) # (batch, heads, seq_len, dim)

    def forward(self, q, k, sequence_length=None):
        if sequence_length is None:
            sequence_length = q.shape[-2]

        # reshape to (1, 1, sequence_length, dim) to braodcast across query and key tensors
        cos = self.cos_cached[:sequence_length, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:sequence_length, :].unsqueeze(0).unsqueeze(0)

        # apply rotaty transformation
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated
