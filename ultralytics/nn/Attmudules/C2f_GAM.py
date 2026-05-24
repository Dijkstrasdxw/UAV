import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.modules.conv import Conv
from .GAM import GAM


class C2f_GAM(nn.Module):
    """C2fGAM block: apply GAM on concatenated features before final 1x1 projection."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, rate: int = 4):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cat_c = (2 + n) * self.c
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.cat_c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        self.gam = GAM(self.cat_c, self.cat_c, rate=rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = torch.cat(y, 1)
        out = self.gam(out)
        return self.cv2(out)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out = torch.cat(y, 1)
        out = self.gam(out)
        return self.cv2(out)
