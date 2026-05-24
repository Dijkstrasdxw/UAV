import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


def _act_layer(name: str):
    """Return activation layer by name."""
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "prelu":
        return nn.PReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "hswish":
        return nn.Hardswish(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise NotImplementedError(f"Unsupported activation: {name}")


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle used in MSCB."""
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(b, c, h, w)


class MSDC(nn.Module):
    """Multi-scale depthwise convolution branches."""

    def __init__(self, c: int, kernel_sizes=(1, 3, 5), stride: int = 1, act: str = "relu6", dw_parallel: bool = True):
        super().__init__()
        self.dw_parallel = bool(dw_parallel)
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, c, k, stride, k // 2, groups=c, bias=False),
                    nn.BatchNorm2d(c),
                    _act_layer(act),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor):
        outs = []
        cur = x
        for dw in self.dwconvs:
            y = dw(cur)
            outs.append(y)
            if not self.dw_parallel:
                cur = cur + y
        return outs


class MSCB(nn.Module):
    """Multi-scale convolution block from EMCAM."""

    def __init__(
        self,
        c1: int,
        c2: int,
        stride: int = 1,
        kernel_sizes=(1, 3, 5),
        expansion_factor: float = 6.0,
        dw_parallel: bool = True,
        add: bool = True,
        act: str = "relu6",
    ):
        super().__init__()
        assert stride in (1, 2)
        self.use_skip = stride == 1
        self.add = bool(add)

        c_mid = int(c1 * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(c1, c_mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_mid),
            _act_layer(act),
        )
        self.msdc = MSDC(c_mid, kernel_sizes=kernel_sizes, stride=stride, act=act, dw_parallel=dw_parallel)

        self.combined_channels = c_mid if self.add else c_mid * len(kernel_sizes)
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.skip_proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if self.use_skip and c1 != c2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pconv1(x)
        ms_out = self.msdc(y)
        if self.add:
            out = 0
            for t in ms_out:
                out = out + t
        else:
            out = torch.cat(ms_out, dim=1)
        out = _channel_shuffle(out, math.gcd(self.combined_channels, self.pconv2[0].out_channels))
        out = self.pconv2(out)

        if self.use_skip:
            skip = x if self.skip_proj is None else self.skip_proj(x)
            out = out + skip
        return out


class CAB(nn.Module):
    """Channel attention block."""

    def __init__(self, c: int, ratio: int = 16, act: str = "relu"):
        super().__init__()
        ratio = max(1, min(ratio, c))
        c_mid = max(1, c // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(c, c_mid, 1, bias=False)
        self.act = _act_layer(act)
        self.fc2 = nn.Conv2d(c_mid, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        m = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return self.sigmoid(a + m)


class SAB(nn.Module):
    """Spatial attention block."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7, 11), "kernel must be 3, 7, or 11"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class EMCAM(nn.Module):
    """EMCAM: CAB + SAB + MSCB."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_sizes=(1, 3, 5),
        expansion_factor: float = 6.0,
        dw_parallel: bool = True,
        add: bool = True,
        act: str = "relu6",
        cab_ratio: int = 16,
        sab_kernel: int = 7,
    ):
        super().__init__()
        self.cab = CAB(c1, ratio=cab_ratio, act="relu")
        self.sab = SAB(kernel_size=sab_kernel)
        self.mscb = MSCB(
            c1,
            c2,
            stride=1,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cab(x) * x
        y = self.sab(y) * y
        return self.mscb(y)


class Bottleneck_EMCAM(nn.Module):
    """Bottleneck using full EMCAM logic."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k=((3, 3), (3, 3)),
        e: float = 0.5,
        kernel_sizes=(1, 3, 5),
        expansion_factor: float = 6.0,
        dw_parallel: bool = True,
        add: bool = True,
        act: str = "relu6",
        cab_ratio: int = 16,
        sab_kernel: int = 7,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, g=g)
        self.emcam = EMCAM(
            c_,
            c_,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            act=act,
            cab_ratio=cab_ratio,
            sab_kernel=sab_kernel,
        )
        self.proj = Conv(c_, c2, k=1, s=1) if c_ != c2 else nn.Identity()
        self.use_shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(self.emcam(self.cv1(x)))
        return x + y if self.use_shortcut else y


class C2f_EMCAM(nn.Module):
    """C2f variant with full EMCAM block in each bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        kernel_sizes=(1, 3, 5),
        expansion_factor: float = 6.0,
        dw_parallel: bool = True,
        add: bool = True,
        act: str = "relu6",
        cab_ratio: int = 16,
        sab_kernel: int = 7,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_EMCAM(
                self.c,
                self.c,
                shortcut=shortcut,
                g=g,
                k=((3, 3), (3, 3)),
                e=1.0,
                kernel_sizes=kernel_sizes,
                expansion_factor=expansion_factor,
                dw_parallel=dw_parallel,
                add=add,
                act=act,
                cab_ratio=cab_ratio,
                sab_kernel=sab_kernel,
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
