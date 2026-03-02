import torch

def gelu(x):
    """
    Gaussian Error Linear Units (GELUs)
    https://arxiv.org/pdf/1606.08415
    """
    return 0.5 * x * (1 + torch.erf(x / (2 ** 0.5)))