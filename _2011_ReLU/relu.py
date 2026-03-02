import torch

def relu(x):
    """
    Sparse Rectifier Neural Networks
    https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdfDeep
    """
    return torch.clamp(x, 0)