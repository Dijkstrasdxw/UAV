"""C2f_PIG and its source-aligned building blocks from PCPE-YOLO."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    """Ensure channels are divisible by a given divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Hard sigmoid used by SE gate."""
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    """SE block used inside GhostBottleneckV2."""

    def __init__(
        self,
        in_chs: int,
        se_ratio: float = 0.25,
        reduced_base_chs: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor: int = 4,
        **_: object,
    ):
        super().__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class GhostModuleV2(nn.Module):
    """GhostModuleV2 from PCPE-YOLO source."""

    def __init__(
        self,
        inp: int,
        oup: int,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        relu: bool = True,
        mode: str | None = None,
        args: object = None,
    ):
        super().__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()
        self.oup = oup

        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        if self.mode == "attn":
            self.short_conv = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        _ = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "attn":
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            gate = F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode="nearest")
            return out[:, : self.oup, :, :] * gate

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class GhostBottleneckV2(nn.Module):
    """GhostBottleneckV2 from PCPE-YOLO source."""

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        dw_kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
        layer_id: int = 0,
        args: object = None,
    ):
        super().__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode="original", args=args)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode="attn", args=args)

        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio) if has_se else None
        self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False, mode="original", args=args)

        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x = x + self.shortcut(residual)
        return x


class PConv(nn.Module):
    """PConv implementation aligned with PCPE-YOLO source (split_cat mode)."""

    def __init__(self, dim: int, ouc: int, n_div: int = 4, forward: str = "split_cat"):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)

        if forward == "slicing":
            self.forward_impl = self.forward_slicing
        elif forward == "split_cat":
            self.forward_impl = self.forward_split_cat
        else:
            raise NotImplementedError(forward)

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return self.conv(x)

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(x)


class InceptionDWConv2d(nn.Module):
    """Inception depthwise conv block used in Bottleneck_PI (source-aligned)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        p: int = 1,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.125,
    ):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(
            gc,
            gc,
            kernel_size=(1, band_kernel_size),
            padding=(0, band_kernel_size // 2),
            groups=gc,
        )
        self.dwconv_h = nn.Conv2d(
            gc,
            gc,
            kernel_size=(band_kernel_size, 1),
            padding=(band_kernel_size // 2, 0),
            groups=gc,
        )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        self.conv = Conv(in_channels, out_channels, square_kernel_size, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)
        return self.conv(x)


class Bottleneck_PI(nn.Module):
    """Bottleneck_PI from PCPE-YOLO source."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[tuple[int, int], tuple[int, int]] = ((3, 3), (3, 3)),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = PConv(c1, c_)
        self.cv2 = InceptionDWConv2d(c_, c2)
        self.add = shortcut and c1 == c2
        _ = (g, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_PIG(nn.Module):
    """Source-aligned C2f_PIG.

    n <= 3: Bottleneck_PI
    n > 3:  GhostBottleneckV2
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        se_ratio: float = 0.0,
        stride: int = 1,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        if n <= 3:
            self.m = nn.ModuleList(
                Bottleneck_PI(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                GhostBottleneckV2(
                    in_chs=self.c,
                    mid_chs=self.c,
                    out_chs=self.c,
                    se_ratio=se_ratio,
                    stride=stride,
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
