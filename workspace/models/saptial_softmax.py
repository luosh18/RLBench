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
        x_map = torch.zeros((n_rows, n_cols), device=device)
        y_map = torch.zeros((n_rows, n_cols), device=device)
        for i in range(n_rows):
            for j in range(n_cols):
                x_map[i, j] = (i - n_rows / 2.0) / n_rows
                y_map[i, j] = (j - n_cols / 2.0) / n_cols
        x_map = x_map.view(n_rows * n_cols, -1)
        y_map = y_map.view(n_rows * n_cols, -1)
        self.x_map = nn.Parameter(x_map, False)
        self.y_map = nn.Parameter(y_map, False)
        self._setup = True

    def forward(self, features: torch.Tensor):
        _, C, W, H = features.shape  # B, C, W, H
        if not self._setup:
            self.setup(W, H, features.device)
        features = features.view(-1, W * H)  # B*C, W*H
        features = F.softmax(features, dim=-1)  # B*C, W*H
        fp_x = torch.sum(
            torch.matmul(features, self.x_map),
            dim=1, keepdim=True
        )  # B*C, 1
        fp_y = torch.sum(
            torch.matmul(features, self.y_map),
            dim=1, keepdim=True
        )  # B*C, 1
        fp_xy = torch.cat((fp_x, fp_y), 1)  # B*C, 2
        features_keypoints = fp_xy.view(-1, C * 2)  # B, C*2
        return features_keypoints


