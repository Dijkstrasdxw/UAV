import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class LADDown(nn.Module):
    """LAD: Light Adaptive-weight Downsampling (LRDS-YOLO paper-aligned implementation).

    Dual-path design:
      1) avg-pool -> 1x1 conv -> spatial softmax attention weights
      2) 3x3 group-conv(stride=2, groups=8) with channel expansion C -> 4C
      3) element-wise modulation: Weights * W
      4) 1x1 projection to target channels for YOLOv8 compatibility

    Notes:
      - The paper text does not provide official code. This implementation follows the published equations.
      - softmax_rescale=False keeps strict paper-style behavior.
      - softmax_rescale=True applies (H*W) gain after softmax for numerical stability experiments.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 8, softmax_rescale: bool = False):
        super().__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.weight_conv = nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0, bias=True)

        g = groups if c1 % groups == 0 else math.gcd(c1, groups)
        g = max(1, g)
        self.group_conv = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, kernel_size=k, stride=2, padding=k // 2, groups=g, bias=False),
            nn.BatchNorm2d(c1 * 4),
            nn.SiLU(inplace=True),
        )

        # Keep output channels compatible with the original backbone stage.
        self.proj = Conv(c1 * 4, c2, k=1, s=1)
        self.softmax_rescale = softmax_rescale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.avg_pool(x)
        z = self.weight_conv(pooled)

        b, _, h, w = z.shape
        weights = F.softmax(z.view(b, 1, -1), dim=2).view(b, 1, h, w)
        if self.softmax_rescale:
            weights = weights * (h * w)

        feat = self.group_conv(x)
        fused = weights * feat
        return self.proj(fused)
