import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """CARAFE upsampler compatible with MMCV CARAFEPack defaults.

    Prefers official mmcv.ops.CARAFEPack when available. Otherwise falls back to
    a pure PyTorch implementation with the same parameter interface.
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        up_kernel: int = 5,
        up_group: int = 1,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        compressed_channels: int = 64,
    ):
        super().__init__()
        if channels % up_group != 0:
            raise ValueError(f"channels ({channels}) must be divisible by up_group ({up_group})")

        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        self.up_kernel = int(up_kernel)
        self.up_group = int(up_group)
        self.encoder_kernel = int(encoder_kernel)
        self.encoder_dilation = int(encoder_dilation)
        self.compressed_channels = int(compressed_channels)

        try:
            from mmcv.ops.carafe import CARAFEPack as MMCV_CARAFEPack
        except Exception:
            MMCV_CARAFEPack = None

        self._mmcv = None
        if MMCV_CARAFEPack is not None:
            self._mmcv = MMCV_CARAFEPack(
                channels=self.channels,
                scale_factor=self.scale_factor,
                up_kernel=self.up_kernel,
                up_group=self.up_group,
                encoder_kernel=self.encoder_kernel,
                encoder_dilation=self.encoder_dilation,
                compressed_channels=self.compressed_channels,
            )
        else:
            self.channel_compressor = nn.Conv2d(self.channels, self.compressed_channels, 1)
            encoder_padding = ((self.encoder_kernel - 1) * self.encoder_dilation) // 2
            self.content_encoder = nn.Conv2d(
                self.compressed_channels,
                self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=encoder_padding,
                dilation=self.encoder_dilation,
            )
            self.init_weights()

    def init_weights(self):
        if self._mmcv is not None:
            return
        nn.init.xavier_uniform_(self.channel_compressor.weight)
        if self.channel_compressor.bias is not None:
            nn.init.zeros_(self.channel_compressor.bias)
        nn.init.normal_(self.content_encoder.weight, std=0.001)
        if self.content_encoder.bias is not None:
            nn.init.zeros_(self.content_encoder.bias)

    def kernel_normalizer(self, mask: torch.Tensor) -> torch.Tensor:
        mask = F.pixel_shuffle(mask, self.scale_factor)
        b, _, h, w = mask.shape
        mask = mask.view(b, self.up_group, self.up_kernel * self.up_kernel, h, w)
        mask = F.softmax(mask, dim=2)
        return mask.view(b, self.up_group * self.up_kernel * self.up_kernel, h, w)

    def feature_reassemble(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale_factor
        k = self.up_kernel
        g = self.up_group

        mask = mask.view(b, g, k * k, h * s, w * s)
        mask = mask.unfold(3, s, s).unfold(4, s, s)
        mask = mask.contiguous().view(b, g, k * k, h, w, s * s)
        mask = mask.permute(0, 1, 3, 4, 2, 5).contiguous()

        x = F.unfold(x, kernel_size=k, dilation=1, padding=k // 2)
        x = x.view(b, g, c // g, k * k, h, w)
        x = x.permute(0, 1, 2, 4, 5, 3).contiguous()

        out = torch.einsum("bgchwk,bghwks->bgchws", x, mask)
        out = out.contiguous().view(b, c, h, w, s, s)
        out = out.permute(0, 1, 2, 4, 3, 5).contiguous()
        return out.view(b, c, h * s, w * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mmcv is not None:
            return self._mmcv(x)
        mask = self.kernel_normalizer(self.content_encoder(self.channel_compressor(x)))
        return self.feature_reassemble(x, mask)
