import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class InceptionMixer(nn.Module):
    """YOLO-friendly InceptionNeXt-style depthwise mixer."""

    def __init__(self, c1: int, c2: int, square_kernel_size: int = 3, band_kernel_size: int = 11, branch_ratio: float = 0.125):
        super().__init__()
        branch_channels = max(int(c1 * branch_ratio), 1)
        branch_channels = min(branch_channels, max((c1 - 1) // 3, 1))
        identity_channels = c1 - 3 * branch_channels

        if identity_channels < 1:
            raise ValueError(f"InceptionMixer requires enough channels, got c1={c1}")

        self.split_indexes = (identity_channels, branch_channels, branch_channels, branch_channels)
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
        self.fuse = Conv(c1, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        return self.fuse(x)


class Bottleneck_Inc(nn.Module):
    """Bottleneck that replaces the second spatial conv with an Inception mixer."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k=((3, 3), (3, 3)),
        e: float = 0.5,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = InceptionMixer(c_, c2, square_kernel_size=k[1][0], band_kernel_size=band_kernel_size, branch_ratio=branch_ratio)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_Inc(nn.Module):
    """C2f variant with Inception-style bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_Inc(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
