import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Mish', 'HardMish', 'Swish']


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class HardMish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (nn.functional.relu(x + 3.) / 3.)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)
