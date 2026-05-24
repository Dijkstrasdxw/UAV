import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from .CA import CA


class BottleneckCA(nn.Module):
    """Bottleneck (with hidden channels) -> CA, no residual."""

    def __init__(self, c: int, g: int = 1, e: float = 1.0):
        super().__init__()
        c_ = int(c * e)  # hidden channels
        # Two 3x3 convs with hidden channels, followed by Coordinate Attention
        self.cv1 = Conv(c, c_, 3, 1)
        self.cv2 = Conv(c_, c, 3, 1, g=g)
        self.attn = CA(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv2(self.cv1(x))
        return self.attn(x)


class C2f_CA(nn.Module):
    """C2f module with CA integrated inside each bottleneck block (no shortcut)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # Per-paper: remove bottleneck residual, integrate CA after bottleneck
        self.m = nn.ModuleList(BottleneckCA(self.c, g=g, e=1.0) for _ in range(n))
        self.shortcut = shortcut  # kept for signature compatibility (unused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
