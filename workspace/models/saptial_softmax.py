import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """B, C, W, H => B, C*2 # for each channel, get feature points [x,y]
    https://github.com/naruya/spatial_softmax-pytorch/blob/master/spatial_softmax.ipynb
    https://blog.csdn.net/weixin_37804469/article/details/108192812
    """

    def __init__(self) -> None:
        super().__init__()
        self._setup = False

    def setup(self, n_rows, n_cols, device=None):
        x_map = np.zeros((n_rows, n_cols))
        y_map = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                x_map[i, j] = (i - n_rows / 2) / n_rows
                y_map[i, j] = (j - n_cols / 2) / n_cols
        x_map = torch.from_numpy(
            np.array(x_map.reshape((-1)), np.float32)).to(device)
        y_map = torch.from_numpy(
            np.array(y_map.reshape((-1)), np.float32)).to(device)

        self.x_map = x_map
        self.y_map = y_map
        self._setup = True

    def forward(self, features: torch.Tensor):
        B, C, W, H = features.shape  # B, C, W, H
        if not self._setup:
            self.setup(W, H, features.device)
        features = features.view(B, C, W*H)  # batch, C, W*H
        features = F.softmax(features, 2)  # batch, C, W*H
        fp_x = torch.matmul(features, self.x_map)  # batch, C
        fp_y = torch.matmul(features, self.y_map)  # batch, C
        features = torch.cat((fp_x, fp_y), 1)
        return features  # batch, C*2
