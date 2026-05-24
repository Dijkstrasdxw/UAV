import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from .PConv import PartialConv3


class FasterBlockPConv(nn.Module):
    """FasterNet-style block: PartialConv3 + PW-MLP."""

    def __init__(
        self,
        c: int,
        shortcut: bool = True,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        pconv_forward: str = "split_cat",
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        c_mid = max(1, int(round(c * mlp_ratio)))
        self.spatial_mixing = PartialConv3(c, n_div=n_div, forward=pconv_forward)
        self.pw1 = nn.Conv2d(c, c_mid, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.act = act_layer(inplace=True) if act_layer in {nn.ReLU, nn.SiLU, nn.LeakyReLU} else act_layer()
        # Official FasterNet MLP second PWConv is linear (no BN/act).
        self.pw2 = nn.Conv2d(c_mid, c, 1, 1, 0, bias=False)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spatial_mixing(x)
        y = self.pw2(self.act(self.bn1(self.pw1(y))))
        return x + y if self.add else y


class C2f_PConv(nn.Module):
    """C2f variant with FasterNet-style inner blocks (paper-friendly migration)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        pconv_forward: str = "split_cat",
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            FasterBlockPConv(
                self.c,
                shortcut=shortcut,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                pconv_forward=pconv_forward,
            )
            for _ in range(n)
        )
        _ = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
