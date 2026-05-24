import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention (EMA)."""

    def __init__(self, channels: int, factor: int = 8):
        super().__init__()
        groups = min(factor, channels)
        while channels % groups != 0:
            groups -= 1
        self.groups = max(groups, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(num_groups=channels // self.groups, num_channels=channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        g = self.groups
        x = x.view(b * g, c // g, h, w)

        x_h = x.mean(dim=3, keepdim=True)  # (b*g, c//g, h, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (b*g, c//g, w, 1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1x1(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x1 = self.gn(x * x_h.sigmoid() * x_w.sigmoid())
        x2 = self.conv3x3(x)

        x11 = self.softmax(self.avg_pool(x1).view(b * g, 1, -1))
        x12 = x2.view(b * g, c // g, -1)
        x21 = self.softmax(self.avg_pool(x2).view(b * g, 1, -1))
        x22 = x1.view(b * g, c // g, -1)
        weights = torch.matmul(x11, x12) + torch.matmul(x21, x22)
        weights = weights.view(b * g, 1, h, w)

        out = x * weights.sigmoid()
        return out.view(b, c, h, w)

