import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class RepLKBlock(nn.Module):
    """Lightweight RepLK-style block with a single large-kernel depthwise path."""

    def __init__(
        self,
        c: int,
        kernel_size: int = 13,
        expand: float = 1.0,
        layer_scale_init: float = 1.0,
    ):
        super().__init__()
        if kernel_size < 5 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd and >= 5, got {kernel_size}")

        c_mid = max(int(round(c * expand)), 1)
        self.pre_norm = nn.BatchNorm2d(c)
        self.pw1 = Conv(c, c_mid, k=1, s=1)
        self.dw = nn.Conv2d(
            c_mid,
            c_mid,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=c_mid,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(c_mid)
        self.act = nn.SiLU(inplace=True)
        self.pw2 = nn.Conv2d(c_mid, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_bn = nn.BatchNorm2d(c)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * float(layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        y = self.pw1(y)
        y = self.act(self.dw_bn(self.dw(y)))
        y = self.out_bn(self.pw2(y))
        return x + self.gamma * y
