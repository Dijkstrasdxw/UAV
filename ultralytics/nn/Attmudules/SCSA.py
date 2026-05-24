"""SCSA attention module adapted for Ultralytics YOLO.

This implementation keeps the official SCSA processing flow:
1. multi-semantic spatial attention on pooled H/W descriptors
2. channel attention via lightweight self-attention on downsampled features
3. reweight the input feature map with the combined attention
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_divisor(value: int, preferred: int) -> int:
    """Pick the largest divisor of value that does not exceed preferred."""
    upper = min(preferred, value)
    for divisor in range(upper, 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


class SCSA(nn.Module):
    """Spatial and Channel Synergistic Attention.

    The block follows the official SCSA design while adapting it to a plain
    PyTorch/Ultralytics environment with no MMEngine or einops dependency.
    """

    def __init__(
        self,
        dim: int,
        head_num: int = 8,
        window_size: int = 7,
        group_kernel_sizes: Sequence[int] = (3, 5, 7, 9),
        qkv_bias: bool = False,
        down_sample_mode: str = "avg_pool",
        gate_layer: str = "sigmoid",
        attn_drop_ratio: float = 0.0,
    ):
        super().__init__()
        if len(group_kernel_sizes) != 4:
            raise ValueError(f"group_kernel_sizes must contain 4 elements, got {group_kernel_sizes}")
        if dim < 4:
            raise ValueError(f"SCSA expects dim >= 4, got {dim}")
        if dim % 4 != 0:
            raise ValueError(f"SCSA expects dim divisible by 4, got {dim}")
        if gate_layer not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported gate_layer={gate_layer}")
        if down_sample_mode not in {"avg_pool", "max_pool", "recombination"}:
            raise ValueError(f"Unsupported down_sample_mode={down_sample_mode}")

        self.dim = dim
        self.window_size = window_size
        self.group_chans = dim // 4
        self.head_num = _pick_divisor(dim, head_num)
        self.head_dim = dim // self.head_num
        self.scaler = self.head_dim ** -0.5
        self.down_sample_mode = down_sample_mode

        k0, k1, k2, k3 = group_kernel_sizes
        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=k0, padding=k0 // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=k1, padding=k1 // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=k2, padding=k2 // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=k3, padding=k3 // 2, groups=self.group_chans)

        gate_cls = nn.Softmax if gate_layer == "softmax" else nn.Sigmoid
        self.sa_gate = gate_cls(dim=2) if gate_layer == "softmax" else gate_cls()
        self.ca_gate = gate_cls(dim=1) if gate_layer == "softmax" else gate_cls()

        # Keep GroupNorm from the official implementation; choose valid group counts for YOLO channels.
        self.norm_h = nn.GroupNorm(_pick_divisor(dim, 4), dim)
        self.norm_w = nn.GroupNorm(_pick_divisor(dim, 4), dim)
        self.norm = nn.GroupNorm(1, dim)

        self.conv_d = nn.Identity()
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        elif down_sample_mode == "avg_pool":
            self.down_func = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        elif down_sample_mode == "max_pool":
            self.down_func = nn.MaxPool2d(kernel_size=window_size, stride=window_size)
        else:
            self.down_func = self.space_to_chans
            self.conv_d = nn.Conv2d(dim * window_size * window_size, dim, kernel_size=1, bias=False)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def space_to_chans(self, x: torch.Tensor) -> torch.Tensor:
        """Recombine spatial windows into channels, padding when needed."""
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h += pad_h
            w += pad_w
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * ws * ws, h // ws, w // ws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # Spatial attention priority calculation.
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(l_x_h),
                        self.global_dwc_s(g_x_h_s),
                        self.global_dwc_m(g_x_h_m),
                        self.global_dwc_l(g_x_h_l),
                    ),
                    dim=1,
                )
            )
        ).view(b, c, h, 1)

        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(l_x_w),
                        self.global_dwc_s(g_x_w_s),
                        self.global_dwc_m(g_x_w_m),
                        self.global_dwc_l(g_x_w_l),
                    ),
                    dim=1,
                )
            )
        ).view(b, c, 1, w)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on lightweight self-attention.
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_ds, w_ds = y.size()

        y = self.norm(y)
        q = self.q(y).reshape(b, self.head_num, self.head_dim, h_ds * w_ds)
        k = self.k(y).reshape(b, self.head_num, self.head_dim, h_ds * w_ds)
        v = self.v(y).reshape(b, self.head_num, self.head_dim, h_ds * w_ds)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = torch.matmul(attn, v).reshape(b, c, h_ds, w_ds)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x
