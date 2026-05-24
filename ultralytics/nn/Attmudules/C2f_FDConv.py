import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, autopad
from .FDConv import FDConv


class FDConvBNAct(nn.Module):
    """FDConv + BN + activation wrapper aligned with Ultralytics Conv usage."""

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k=3, s=1, p=None, g: int = 1, d: int = 1, act=True, **fd_kwargs):
        super().__init__()
        self.conv = FDConv(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
            **fd_kwargs,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if y.dtype != self.bn.weight.dtype:
            y = y.to(self.bn.weight.dtype)
        return self.act(self.bn(y))


class Bottleneck_FDConv(nn.Module):
    """Standard Bottleneck with FDConv as the second 3x3 convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k=((3, 3), (3, 3)),
        e: float = 0.5,
        kernel_num: int = 4,
        **fd_kwargs,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = FDConvBNAct(c_, c2, k[1], 1, g=g, kernel_num=kernel_num, **fd_kwargs)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_FDConv(nn.Module):
    """C2f block whose inner bottlenecks use FDConv for spatial extraction."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        kernel_num: int = 4,
        **fd_kwargs,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_FDConv(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
                kernel_num=kernel_num,
                **fd_kwargs,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
