import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class TRDDown(nn.Module):
    """Texture Reconstructive Downsampling (TRD).

    Paper-aligned core path:
      1) baseline stride-2 convolution,
      2) inverse projection to input resolution,
      3) residual extraction (input - projected baseline),
      4) average pooling (k=3, s=2, p=1),
      5) 1x1 reconstruction and additive fusion.

    Args:
        c1 (int): input channels.
        c2 (int): output channels.
        lite (bool): fast approximation mode. Keeps the same module interface used in existing YAMLs.
    """

    def __init__(self, c1: int, c2: int, lite: bool = False):
        super().__init__()
        self.lite = lite

        # Baseline downsampling branch.
        self.down = Conv(c1, c2, k=3, s=2)

        # Channel alignment before residual extraction.
        self.align = Conv(c1, c2, k=1, s=1, act=False)

        # Reconstruction projection after residual pooling.
        self.res_pw = Conv(c2, c2, k=1, s=1, act=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.down(x)
        x_aligned = self.align(x)

        if self.lite:
            # Fast approximation: avoid inverse projection for better throughput.
            residual = F.avg_pool2d(x_aligned, kernel_size=3, stride=2, padding=1) - base
        else:
            # Paper-like inverse projection and residual reconstruction.
            base_up = F.interpolate(base, size=x_aligned.shape[-2:], mode="bilinear", align_corners=False)
            residual = F.avg_pool2d(x_aligned - base_up, kernel_size=3, stride=2, padding=1)

        out = base + self.res_pw(residual)
        return self.act(out)
