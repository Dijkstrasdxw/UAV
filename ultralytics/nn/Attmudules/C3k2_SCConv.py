import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import Bottleneck, C2f, C3
from ultralytics.nn.modules.conv import Conv


def _valid_groups(channels: int, max_groups: int) -> int:
    groups = max(1, min(int(max_groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class SRU(nn.Module):
    """Spatial Reconstruction Unit from SCConv."""

    def __init__(self, channels: int, group_num: int = 4, gate_threshold: float = 0.5):
        super().__init__()
        self.gn = nn.GroupNorm(_valid_groups(channels, group_num), channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def reconstruct(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x11, x12 = torch.chunk(x1, 2, dim=1)
        x21, x22 = torch.chunk(x2, 2, dim=1)
        return torch.cat((x11 + x22, x12 + x21), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gn_x = self.gn(x)
        gamma = self.gn.weight / (self.gn.weight.sum() + 1e-6)
        gamma = gamma.view(1, -1, 1, 1)
        reweights = self.sigmoid(gn_x * gamma)
        w1 = torch.where(reweights > self.gate_threshold, torch.ones_like(reweights), reweights)
        w2 = torch.where(reweights > self.gate_threshold, torch.zeros_like(reweights), reweights)
        return self.reconstruct(w1 * x, w2 * x)


class CRU(nn.Module):
    """Channel Reconstruction Unit from SCConv."""

    def __init__(
        self,
        channels: int,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()
        up_channel = max(2, int(alpha * channels))
        low_channel = max(2, channels - up_channel)
        if up_channel % 2:
            up_channel -= 1
            low_channel += 1
        if low_channel % 2:
            low_channel -= 1
            up_channel += 1

        self.up_channel = up_channel
        self.low_channel = low_channel

        up_squeezed = max(1, up_channel // squeeze_ratio)
        low_squeezed = max(1, low_channel // squeeze_ratio)
        groups = _valid_groups(up_squeezed, group_size)

        self.squeeze1 = nn.Conv2d(up_channel, up_squeezed, 1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_squeezed, 1, bias=False)
        self.gwc = nn.Conv2d(
            up_squeezed,
            channels,
            kernel_size=group_kernel_size,
            stride=1,
            padding=group_kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.pwc1 = nn.Conv2d(up_squeezed, channels, 1, bias=False)
        self.pwc2 = nn.Conv2d(low_squeezed, channels - low_squeezed, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up = self.squeeze1(up)
        low = self.squeeze2(low)

        y1 = self.gwc(up) + self.pwc1(up)
        y2 = torch.cat((self.pwc2(low), low), dim=1)

        out = torch.cat((y1, y2), dim=1)
        out = F.softmax(self.pool(out), dim=1) * out
        out1, out2 = torch.chunk(out, 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    """Spatial and channel reconstruction convolution."""

    def __init__(
        self,
        channels: int,
        group_num: int = 4,
        gate_threshold: float = 0.5,
        alpha: float = 0.5,
        squeeze_ratio: int = 2,
        group_size: int = 2,
        group_kernel_size: int = 3,
    ):
        super().__init__()
        self.sru = SRU(channels, group_num=group_num, gate_threshold=gate_threshold)
        self.cru = CRU(
            channels,
            alpha=alpha,
            squeeze_ratio=squeeze_ratio,
            group_size=group_size,
            group_kernel_size=group_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cru(self.sru(x))


class Bottleneck_SCConv(nn.Module):
    """Half-replacement bottleneck: first Conv stays vanilla, second Conv becomes ScConv."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 1.0
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c_)
        self.add = shortcut and c1 == c2 and c_ == c2
        _ = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k_SCConv(C3):
    """C3k variant whose inner bottlenecks use ScConv."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SCConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_SCConv(C2f):
    """Half-replacement C3k2: only c3k=True branch uses ScConv-enhanced C3k."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_SCConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
