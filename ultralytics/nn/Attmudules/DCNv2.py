import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ultralytics.nn.modules.conv import autopad

try:
    from torchvision.ops import deform_conv2d
except Exception as e:  # pragma: no cover
    deform_conv2d = None
    _TORCHVISION_IMPORT_ERROR = e


class DCNv2(nn.Module):
    """Modulated Deformable Conv v2 using torchvision.ops.deform_conv2d."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k=3,
        s=1,
        p=None,
        g: int = 1,
        d: int = 1,
        offset_groups: int = 1,
        modulated: bool = True,
        bias: bool = False,
        act=True,
    ):
        super().__init__()
        if c1 % g != 0 or c2 % g != 0:
            raise ValueError(f"DCNv2 groups mismatch: c1={c1}, c2={c2}, g={g}")
        if c1 % offset_groups != 0:
            raise ValueError(f"DCNv2 offset_groups mismatch: c1={c1}, offset_groups={offset_groups}")

        self.in_channels = c1
        self.out_channels = c2
        self.groups = g
        self.offset_groups = offset_groups
        self.modulated = modulated

        self.kernel_size = _pair(k)
        self.stride = _pair(s)
        self.dilation = _pair(d)
        self.padding = _pair(autopad(k, p, d))
        kh, kw = self.kernel_size

        self.weight = nn.Parameter(torch.empty(c2, c1 // g, kh, kw))
        self.bias = nn.Parameter(torch.zeros(c2)) if bias else None

        offset_mask_channels = offset_groups * kh * kw * (3 if modulated else 2)
        self.conv_offset_mask = nn.Conv2d(
            c1,
            offset_mask_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
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
                "torchvision.ops.deform_conv2d is required for DCNv2. "
                "Install a compatible torchvision build."
            ) from _TORCHVISION_IMPORT_ERROR

        if not x.is_cuda:
            y = F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            return self.act(self.bn(y))

        x = x.contiguous()
        offset_mask = self.conv_offset_mask(x).contiguous()
        kh, kw = self.kernel_size
        offset_channels = 2 * self.offset_groups * kh * kw
        offset = offset_mask[:, :offset_channels, :, :].contiguous()
        mask = (
            torch.sigmoid(offset_mask[:, offset_channels:, :, :].contiguous())
            if self.modulated
            else None
        )

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
        except RuntimeError as e:
            if not lowp_input or not use_native_dtype:
                raise
            self._force_fp32_lowp = True
            if not self._warned_fp32_fallback:
                warnings.warn(
                    "DCNv2 native low-precision path is unavailable on this environment; falling back to fp32.",
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

        return self.act(self.bn(y))
