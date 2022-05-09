from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, p=2.0, dim=1, eps=1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, input):
        F.normalize(input, self.p, self.dim, self.eps)
