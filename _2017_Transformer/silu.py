from _1986_Backpropagation.sigmoid import sigmoid

def silu(x):
    """
    Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
    https://arxiv.org/pdf/1702.03118
    """
    return x * sigmoid(x)