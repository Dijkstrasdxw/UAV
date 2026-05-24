import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_haar_filters(channels: int, dtype: torch.dtype = torch.float32):
    """Create fixed 2D Haar analysis filters for grouped wavelet downsampling."""
    s = 1.0 / math.sqrt(2.0)
    lo = torch.tensor([s, s], dtype=dtype)
    hi = torch.tensor([s, -s], dtype=dtype)
    filters = torch.stack(
        [
            lo.unsqueeze(0) * lo.unsqueeze(1),
            lo.unsqueeze(0) * hi.unsqueeze(1),
            hi.unsqueeze(0) * lo.unsqueeze(1),
            hi.unsqueeze(0) * hi.unsqueeze(1),
        ],
        dim=0,
    )
    return filters[:, None].repeat(channels, 1, 1, 1)


class HWD(nn.Module):
    """
    Haar Wavelet Downsampling.

    This is a dependency-free YOLO adaptation of the common HWD block:
    - fixed Haar wavelet decomposition with stride-2
    - concatenate four sub-bands along channels
    - 1x1 projection to the target width
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.register_buffer("wt_filter", _create_haar_filters(c1), persistent=False)
        self.project = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if (h % 2) or (w % 2):
            x = F.pad(x, (0, w % 2, 0, h % 2))

        b, c, _, _ = x.shape
        y = F.conv2d(x, self.wt_filter, stride=2, groups=c, padding=0)
        y = y.reshape(b, c, 4, y.shape[-2], y.shape[-1])

        # Keep the classic four-band layout expected by HWD-style modules.
        y = torch.cat([y[:, :, i, :, :] for i in range(4)], dim=1)
        return self.project(y)
