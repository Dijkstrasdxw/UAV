import torch
import torch.nn as nn


def act_layer(act: str, inplace: bool = True, neg_slope: float = 0.2, n_prelu: int = 1) -> nn.Module:
    """Build activation layer by name."""
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=inplace)
    if act == "relu6":
        return nn.ReLU6(inplace=inplace)
    if act == "leakyrelu":
        return nn.LeakyReLU(neg_slope, inplace=inplace)
    if act == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == "gelu":
        return nn.GELU()
    if act == "hswish":
        return nn.Hardswish(inplace=inplace)
    raise NotImplementedError(f"activation layer [{act}] is not found")


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle used in PCPE EUCB."""
    b, c, h, w = x.shape
    if c % groups != 0:
        return x
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class EUCB(nn.Module):
    """
    Efficient Up-Convolution Block from PCPE-YOLO.

    Structure:
    Upsample(2x) -> DWConv3x3 -> BN -> Act -> ChannelShuffle -> PWConv1x1
    """

    def __init__(self, in_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True),
        )
        self.pwc = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
