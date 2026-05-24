import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ultralytics.nn.modules.conv import Conv, autopad

try:
    from torchvision.ops import deform_conv2d
except Exception as e:  # pragma: no cover
    deform_conv2d = None
    _TORCHVISION_IMPORT_ERROR = e


class DeformConv(nn.Module):
    """Self-contained deformable conv wrapper for DLKA."""

    def __init__(
        self,
        channels: int,
        kernel_size=(3, 3),
        padding=None,
        stride=1,
        dilation=1,
        groups=None,
        offset_groups=1,
        modulated=False,
        bias=False,
    ):
        super().__init__()
        groups = channels if groups is None else groups
        self.in_channels = channels
        self.out_channels = channels
        self.groups = groups
        self.offset_groups = offset_groups
        self.modulated = modulated
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = _pair(autopad(kernel_size, padding, dilation))
        kh, kw = self.kernel_size

        if channels % groups != 0:
            raise ValueError(f"DeformConv groups mismatch: channels={channels}, groups={groups}")
        if channels % offset_groups != 0:
            raise ValueError(f"DeformConv offset_groups mismatch: channels={channels}, offset_groups={offset_groups}")

        self.weight = nn.Parameter(torch.empty(channels, channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        offset_mask_channels = offset_groups * kh * kw * (3 if modulated else 2)
        self.conv_offset_mask = nn.Conv2d(
            channels,
            offset_mask_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self._force_fp32_lowp = False
        self._warned_fp32_fallback = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.conv_offset_mask.weight, 0.0)
        nn.init.constant_(self.conv_offset_mask.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if deform_conv2d is None:  # pragma: no cover
            raise ImportError(
                "torchvision.ops.deform_conv2d is required for DLKA. "
                "Install a compatible torchvision build."
            ) from _TORCHVISION_IMPORT_ERROR

        if not x.is_cuda:
            offset_mask = self.conv_offset_mask(x)
            if offset_mask.shape[-2:] != x.shape[-2:]:
                x = F.interpolate(x, size=offset_mask.shape[-2:], mode="nearest")
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        x = x.contiguous()
        offset_mask = self.conv_offset_mask(x).contiguous()
        kh, kw = self.kernel_size
        offset_channels = 2 * self.offset_groups * kh * kw
        offset = offset_mask[:, :offset_channels, :, :].contiguous()
        mask = torch.sigmoid(offset_mask[:, offset_channels:, :, :].contiguous()) if self.modulated else None

        lowp_input = x.dtype in (torch.float16, torch.bfloat16)
        use_native_dtype = not (lowp_input and self._force_fp32_lowp)

        try:
            y = deform_conv2d(
                input=x if use_native_dtype else x.float(),
                offset=offset if use_native_dtype else offset.float(),
                weight=self.weight if use_native_dtype else self.weight.float(),
                bias=self.bias if use_native_dtype or self.bias is None else self.bias.float(),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask if use_native_dtype or mask is None else mask.float(),
            )
        except RuntimeError:
            if not lowp_input or not use_native_dtype:
                raise
            self._force_fp32_lowp = True
            if not self._warned_fp32_fallback:
                warnings.warn(
                    "DLKA DeformConv low-precision path is unavailable; falling back to fp32.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_fp32_fallback = True
            y = deform_conv2d(
                input=x.float(),
                offset=offset.float(),
                weight=self.weight.float(),
                bias=self.bias.float() if self.bias is not None else None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask.float() if mask is not None else None,
            )
            y = y.to(x.dtype)
        return y


class _DeformableLKA(nn.Module):
    """Deformable large-kernel spatial gating unit."""

    def __init__(self, channels: int, k1: int = 5, k2: int = 7, dilation: int = 3):
        super().__init__()
        p1 = (k1 - 1) // 2
        p2 = ((k2 - 1) * dilation) // 2
        self.conv0 = DeformConv(channels, kernel_size=(k1, k1), padding=p1, groups=channels, modulated=False)
        self.conv_spatial = DeformConv(
            channels,
            kernel_size=(k2, k2),
            padding=p2,
            dilation=dilation,
            groups=channels,
            modulated=False,
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class DLKA(nn.Module):
    """Deformable large-kernel attention block with residual projection."""

    def __init__(self, channels: int, k1: int = 5, k2: int = 7, dilation: int = 3, expand: float = 1.0):
        super().__init__()
        mid_channels = max(1, int(round(channels * expand)))
        self.proj_in = Conv(channels, mid_channels, 1, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = _DeformableLKA(mid_channels, k1=k1, k2=k2, dilation=dilation)
        self.proj_out = Conv(mid_channels, channels, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        y = self.proj_in(x)
        y = self.act(y)
        y = self.spatial_gating_unit(y)
        y = self.proj_out(y)
        return y + shortcut
