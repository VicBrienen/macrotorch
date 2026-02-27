import torch


def relu(x):
    """
    Sparse Rectifier Neural Networks
    https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdfDeep
    """
    return torch.clamp(x, 0)

def gelu(x):
    """
    Gaussian Error Linear Units (GELUs)
    https://arxiv.org/pdf/1606.08415
    """
    return 0.5 * x * (1 + torch.erf(x / (2 ** 0.5)))

def sigmoid(x):
    """
    Learning representations by back-propagating errors
    https://www.nature.com/articles/323533a0
    """
    return 1 / (1 + torch.exp(-x))

def silu(x):
    """
    Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
    https://arxiv.org/pdf/1702.03118
    """
    return x * sigmoid(x)

def glu(a, b, gate_fn):
    """
    Language Modeling with Gated Convolutional Networks
    https://arxiv.org/pdf/1612.08083
    """
    return a * gate_fn(b)