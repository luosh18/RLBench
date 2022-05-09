from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorNorm(nn.Module):
    def __init__(self, ord=2):
        super().__init__()
        self.ord = ord

    def forward(self, input: torch.Tensor):
        input = input.view(input.shape[0], -1)  # [B, -1]
        input = torch.linalg.vector_norm(input, ord=self.ord, dim=1)  # [B]
        return input
