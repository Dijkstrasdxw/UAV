import torch
import torch.nn as nn


class DSConv7x7(nn.Module):
    """Depthwise-separable 7x7 conv block used to lighten GAM spatial attention."""

    def __init__(self, c1: int, c2: int, act: bool = True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, kernel_size=7, stride=1, padding=3, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class GAM(nn.Module):
    """
    Global Attention Mechanism (GAM).

    Reference implementation style:
    - channel attention with 2-layer MLP on per-location channel vectors
    - spatial attention with depthwise-separable 7x7 conv blocks
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, rate: int = 4):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden = max(1, in_channels // rate)

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=True),
        )

        self.spatial_attention = nn.Sequential(
            DSConv7x7(in_channels, hidden, act=True),
            DSConv7x7(hidden, out_channels, act=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_perm = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        ch_att = self.act(self.channel_attention(x_perm)).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x * ch_att
        sp_att = self.act(self.spatial_attention(x))
        return x * sp_att
