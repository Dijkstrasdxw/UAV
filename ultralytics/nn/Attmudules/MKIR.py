import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle used after concatenating multi-kernel depthwise branches."""
    if groups <= 1:
        return x
    b, c, h, w = x.shape
    if c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class MultiKernelDepthwiseConv(nn.Module):
    """Parallel multi-kernel depthwise convolutions following the MKDC idea."""

    def __init__(self, channels: int, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([Conv(channels, channels, k=k, s=1, g=channels) for k in kernel_sizes])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [branch(x) for branch in self.branches]


class MKIR(nn.Module):
    """YOLO-adapted Multi-Kernel Inverted Residual block inspired by MK-UNet."""

    def __init__(
        self,
        c1: int,
        c2: int,
        expansion: float = 2.0,
        kernel_sizes=(1, 3, 5),
        add: bool = False,
        shortcut: bool = True,
    ):
        super().__init__()
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        hidden = max(int(c1 * expansion), 1)
        self.add = add
        self.shortcut = shortcut and c1 == c2

        self.expand = Conv(c1, hidden, k=1, s=1)
        self.mkdc = MultiKernelDepthwiseConv(hidden, self.kernel_sizes)
        fused_channels = hidden if self.add else hidden * len(self.kernel_sizes)
        self.project = Conv(fused_channels, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        branches = self.mkdc(y)
        if self.add:
            fused = branches[0]
            for branch in branches[1:]:
                fused = fused + branch
        else:
            fused = torch.cat(branches, dim=1)
            fused = channel_shuffle(fused, math.gcd(fused.shape[1], self.project.conv.out_channels))
        y = self.project(fused)
        return x + y if self.shortcut else y
