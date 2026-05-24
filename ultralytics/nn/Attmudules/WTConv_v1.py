import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_haar_filters(channels: int, dtype: torch.dtype = torch.float32):
    """Create fixed 2D Haar wavelet analysis/synthesis filters."""
    s = 1.0 / math.sqrt(2.0)
    lo = torch.tensor([s, s], dtype=dtype)
    hi = torch.tensor([s, -s], dtype=dtype)

    dec_filters = torch.stack(
        [
            lo.unsqueeze(0) * lo.unsqueeze(1),
            lo.unsqueeze(0) * hi.unsqueeze(1),
            hi.unsqueeze(0) * lo.unsqueeze(1),
            hi.unsqueeze(0) * hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = dec_filters.clone()

    dec_filters = dec_filters[:, None].repeat(channels, 1, 1, 1)
    rec_filters = rec_filters[:, None].repeat(channels, 1, 1, 1)
    return dec_filters, rec_filters


def _wavelet_2d_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = F.conv2d(x, filters, stride=2, groups=c, padding=0)
    return x.reshape(b, c, 4, h // 2, w // 2)


def _inverse_2d_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    b, c, _, h_half, w_half = x.shape
    x = x.reshape(b, c * 4, h_half, w_half)
    return F.conv_transpose2d(x, filters, stride=2, groups=c, padding=0)


class _Scale(nn.Module):
    def __init__(self, channels: int, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _WTConv2d(nn.Module):
    """Pure-PyTorch WTConv adapted from the official repo, using fixed Haar filters only."""

    def __init__(self, channels: int, kernel_size: int = 5, wt_levels: int = 1, bias: bool = False):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if wt_levels < 1:
            raise ValueError(f"wt_levels must be >= 1, got {wt_levels}")

        self.channels = channels
        self.wt_levels = wt_levels

        wt_filter, iwt_filter = _create_haar_filters(channels)
        self.register_buffer("wt_filter", wt_filter, persistent=False)
        self.register_buffer("iwt_filter", iwt_filter, persistent=False)

        self.base_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=bias,
        )
        self.base_scale = _Scale(channels, init_scale=1.0)

        self.wavelet_convs = nn.ModuleList(
            nn.Conv2d(
                channels * 4,
                channels * 4,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels * 4,
                bias=False,
            )
            for _ in range(wt_levels)
        )
        self.wavelet_scales = nn.ModuleList(_Scale(channels * 4, init_scale=0.1) for _ in range(wt_levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2) or (curr_shape[3] % 2):
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))

            curr_x = _wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scales[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0
        for _ in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = _inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        wavelet_out = next_x_ll
        base_out = self.base_scale(self.base_conv(x))
        return base_out + wavelet_out


class _ChannelGate(nn.Module):
    """Lightweight SE-style gate: attention only serves as a usage controller."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // max(int(reduction), 1), 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))


class WTConv(nn.Module):
    """
    YOLO-adapted WTConv block for detect-only P2 refinement.

    Logic:
    - WTConv expands effective receptive field.
    - A very light channel gate only modulates how much large-RF output to use.
    - Residual interpolation preserves original P2 detail if the large-RF branch is not helpful.
    """

    def __init__(self, c: int, kernel_size: int = 5, wt_levels: int = 1, gate_reduction: int = 8):
        super().__init__()
        self.wtconv = _WTConv2d(c, kernel_size=kernel_size, wt_levels=wt_levels, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.gate = _ChannelGate(c, reduction=gate_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.bn(self.wtconv(x))
        gate = self.gate(x)
        return x + gate * (enhanced - x)
