"""
Learning representations by back-propagating errors
https://www.nature.com/articles/323533a0
"""

import torch

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))