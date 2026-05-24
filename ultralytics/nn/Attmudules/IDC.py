import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class InceptionDWConv(nn.Module):
    """Strip-kernel depthwise mixer from IDC/InceptionNeXt style blocks."""

    def __init__(
        self,
        channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ):
        super().__init__()
        branch_channels = max(int(channels * branch_ratio), 1)
        if channels - 3 * branch_channels <= 0:
            branch_channels = max(channels // 8, 1)
            while channels - 3 * branch_channels <= 0 and branch_channels > 1:
                branch_channels -= 1

        self.split_indexes = (channels - 3 * branch_channels, branch_channels, branch_channels, branch_channels)
        self.dwconv_hw = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=square_kernel_size,
            stride=1,
            padding=square_kernel_size // 2,
            groups=branch_channels,
            bias=False,
        )
        self.dwconv_w = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(1, band_kernel_size),
            stride=1,
            padding=(0, band_kernel_size // 2),
            groups=branch_channels,
            bias=False,
        )
        self.dwconv_h = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(band_kernel_size, 1),
            stride=1,
            padding=(band_kernel_size // 2, 0),
            groups=branch_channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class IDCBlock(nn.Module):
    """YOLO-friendly IDC block: 1x1 expand -> strip-kernel DW mixer -> 1x1 project -> residual."""

    def __init__(
        self,
        c1: int,
        c2: int,
        expansion: float = 2.0,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
        shortcut: bool = True,
    ):
        super().__init__()
        c_mid = max(int(c1 * expansion), 1)
        self.shortcut = shortcut and c1 == c2
        self.expand = Conv(c1, c_mid, k=1, s=1)
        self.mixer = InceptionDWConv(
            c_mid,
            square_kernel_size=square_kernel_size,
            band_kernel_size=band_kernel_size,
            branch_ratio=branch_ratio,
        )
        self.project = Conv(c_mid, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.mixer(self.expand(x)))
        return x + y if self.shortcut else y
