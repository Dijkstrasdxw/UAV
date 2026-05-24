import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.amax(x, dim=1, keepdim=True), torch.mean(x, dim=1, keepdim=True)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.conv(self.compress(x)))
        return x * scale


class TripletAttention(nn.Module):
    """Triplet Attention with channel-width, height-channel and spatial branches."""

    def __init__(self, channels: int | None = None, no_spatial: bool = False, kernel_size: int = 7):
        super().__init__()
        self.cw = AttentionGate(kernel_size)
        self.hc = AttentionGate(kernel_size)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1).permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2).permute(0, 3, 2, 1).contiguous()

        if self.no_spatial:
            return 0.5 * (x_out1 + x_out2)

        x_out3 = self.hw(x)
        return (x_out1 + x_out2 + x_out3) / 3.0
