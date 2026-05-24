import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv
from .EMA import EMA


class C2f_EMA(nn.Module):
    """C2f module with embedded EMA attention on a selected bottleneck block."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        self.ema = EMA(self.c)
        self.ema_index = 1 if n > 1 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for i, m in enumerate(self.m):
            if i == self.ema_index:
                y.append(self.ema(m(y[-1])))
            else:
                y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for i, m in enumerate(self.m):
            if i == self.ema_index:
                y.append(self.ema(m(y[-1])))
            else:
                y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))
