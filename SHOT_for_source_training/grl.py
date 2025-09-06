import torch
from torch.autograd import Function
import torch.nn as nn



class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradientReverseFunction.apply(x, lambd)

class GradientReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.lambda_)

class WarmStartGradientReverseLayer(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return grad_reverse(x, self.alpha)
