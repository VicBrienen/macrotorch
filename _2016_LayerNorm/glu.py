import torch

def glu(a, b, gate_fn):
    """
    Language Modeling with Gated Convolutional Networks
    https://arxiv.org/pdf/1612.08083
    """
    return a * gate_fn(b)