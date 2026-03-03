"""
Adam: A Method for Stochastic Optimization
https://arxiv.org/pdf/1412.6980
"""

import torch
from torch.optim import Optimizer

class Adam(Optimizer):
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8):

        # store hyperparameters in dictionary and use it to initialize parameter group
        # enables different sets of hyperparameters per parameter group
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self):

        # iterate over parameter groups
        for group in self.param_groups:
            b1, b2 = group["betas"]

            # iterate over tensors in group
            for param in group["params"]:
                if param.grad is None:
                    continue

                # get or initialize the optimizer state
                state = self.state[param]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                # increment timestep counter and extract state variables
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]

                # update moving average of gradients
                # m = b1 * m + (1 - b1) * grad
                m.mul_(b1).add_(param.grad, alpha = 1 - b1) # we use operation methods instead of regular operations to avoid creating new tensors

                # update moving average of squared gradients
                # v = b2 * v + (1 - b2) * grad ** 2
                v.mul_(b2).addcmul_(param.grad, param.grad, value = 1 - b2)

                # correct for bias towards zero because that is their starting point
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)

                # gradient descent step 
                denominator = v_hat.sqrt().add(group["eps"]) # denominator = sqrt(v_hat) + eps
                param.addcdiv_(m_hat, denominator, value=-group["lr"]) # parameter = parameter - (lr * m_hat / denominator)