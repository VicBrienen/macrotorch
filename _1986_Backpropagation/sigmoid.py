import torch

def sigmoid(x):
    """
    Learning representations by back-propagating errors
    https://www.nature.com/articles/323533a0
    """
    return 1 / (1 + torch.exp(-x))