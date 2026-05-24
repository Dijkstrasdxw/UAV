import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLCA(nn.Module):
    """Mixed Local Channel Attention."""

    def __init__(self, in_size: int, local_size: int = 5, gamma: int = 2, b: int = 1, local_weight: float = 0.5):
        super().__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        self.local_weight = local_weight

        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1
        k = max(k, 1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, m, n = x.shape
        _, c_local, _, _ = local_arv.shape

        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(
            b, c, self.local_size, self.local_size
        )
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)

        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(
            att_global * (1 - self.local_weight) + att_local * self.local_weight,
            [m, n],
        )
        return x * att_all
