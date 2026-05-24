import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from .RFAConv import RFAConv


class BottleneckRFA(nn.Module):
    """Bottleneck with RFAConv replacing the 3x3 conv (stride=1)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        # Replace both 3x3 convs with RFAConv (stride=1) to match C2f(Bottleneck) k=(3,3)
        self.cv1 = RFAConv(c1, c_, kernel_size=3, stride=1)
        self.cv2 = RFAConv(c_, c2, kernel_size=3, stride=1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_RFA(nn.Module):
    """C2f module that uses BottleneckRFA inside (RFAConv on 3x3)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(BottleneckRFA(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
