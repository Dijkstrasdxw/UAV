import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class MultiScaleDWConv(nn.Module):
    """Lightweight multi-scale depthwise block with learnable branch weighting."""

    def __init__(self, c: int, dilations=(1, 2, 4), weighted: bool = True):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.weighted = bool(weighted)
        self.branches = nn.ModuleList([Conv(c, c, k=3, s=1, d=d, g=c) for d in self.dilations])
        self.w = nn.Parameter(torch.ones(len(self.dilations), dtype=torch.float32), requires_grad=True)
        self.pw = Conv(c, c, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [m(x) for m in self.branches]
        if self.weighted:
            w = torch.softmax(self.w, dim=0)
            y = sum(w[i] * ys[i] for i in range(len(ys)))
        else:
            y = sum(ys) / len(ys)
        return self.pw(y)


class Bottleneck_MSDW(nn.Module):
    """Bottleneck with multi-scale depthwise fusion replacing the second 3x3 conv."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k=((3, 3), (3, 3)),
        e: float = 0.5,
        dilations=(1, 2, 4),
        weighted: bool = True,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = MultiScaleDWConv(c_, dilations=dilations, weighted=weighted)
        self.proj = Conv(c_, c2, k=1, s=1) if c_ != c2 else nn.Identity()
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class C2f_MSDW(nn.Module):
    """C2f variant using Bottleneck_MSDW for multi-scale receptive field modeling."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        dilations=(1, 2, 4),
        weighted: bool = True,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_MSDW(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
                dilations=dilations,
                weighted=weighted,
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
