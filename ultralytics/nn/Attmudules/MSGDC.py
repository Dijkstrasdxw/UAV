import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class MSGDC(nn.Module):
    """YOLO-friendly multi-scale grouped dilated convolution block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 1.0,
        dilations=(1, 2, 4),
        group_base: int = 4,
        shortcut: bool = True,
    ):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        hidden = int(c2 * e)
        hidden = max(hidden, 1)

        self.pre = Conv(c1, hidden, k=1, s=1)
        branch_groups = math.gcd(hidden, int(group_base))
        branch_groups = max(branch_groups, 1)
        self.branches = nn.ModuleList([Conv(hidden, hidden, k=3, s=1, d=d, g=branch_groups) for d in self.dilations])
        self.fuse = Conv(hidden * len(self.dilations), c2, k=1, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        y = self.fuse(torch.cat([branch(y) for branch in self.branches], dim=1))
        return x + y if self.shortcut else y
