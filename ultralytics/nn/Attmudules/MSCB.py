import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation for MSCBBlock: {name}")


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle channels after branch fusion."""
    if groups <= 1:
        return x
    b, c, h, w = x.shape
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class MSDCBlock(nn.Module):
    """Multi-scale depthwise convolution branches."""

    def __init__(self, channels: int, kernel_sizes=(1, 3, 5), stride: int = 1, act: str = "relu6", dw_parallel: bool = True):
        super().__init__()
        self.dw_parallel = bool(dw_parallel)
        self.branches = nn.ModuleList(
            [Conv(channels, channels, k=k, s=stride, g=channels, act=_make_activation(act)) for k in kernel_sizes]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        current = x
        for branch in self.branches:
            y = branch(current)
            outs.append(y)
            if not self.dw_parallel:
                current = current + y
        return outs


class MSCBBlock(nn.Module):
    """Standalone MSCB without the extra EMCAM attention wrappers."""

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 1,
        kernel_sizes=(1, 3, 5),
        expansion_factor: float = 2.0,
        dw_parallel: bool = True,
        add: bool = True,
        shortcut: bool = True,
        act: str = "relu6",
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"MSCBBlock only supports stride 1 or 2, got {stride}")

        c_mid = max(int(c1 * expansion_factor), 1)
        self.add = bool(add)
        self.shortcut = shortcut and stride == 1 and c1 == c2
        self.expand = Conv(c1, c_mid, k=1, s=1, act=_make_activation(act))
        self.msdc = MSDCBlock(c_mid, kernel_sizes=kernel_sizes, stride=stride, act=act, dw_parallel=dw_parallel)
        fused_channels = c_mid if self.add else c_mid * len(kernel_sizes)
        self.project = Conv(fused_channels, c2, k=1, s=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        branches = self.msdc(y)
        if self.add:
            fused = 0
            for branch in branches:
                fused = fused + branch
        else:
            fused = torch.cat(branches, dim=1)
        fused = _channel_shuffle(fused, math.gcd(fused.shape[1], self.project.conv.out_channels))
        y = self.project(fused)
        return x + y if self.shortcut else y
