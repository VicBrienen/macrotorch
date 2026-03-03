"""
Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
https://arxiv.org/pdf/1702.03118
"""

from _1986_Backpropagation.sigmoid import sigmoid

def silu(x):
    return x * sigmoid(x)