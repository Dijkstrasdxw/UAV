"""CAA attention module (Context Anchor Attention) from PCPE-YOLO."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class CAA(nn.Module):
    """Context Anchor Attention.

    Applies local average pooling and two strip depthwise convolutions to generate
    an attention factor, then reweights the input feature map.
    """

    def __init__(self, ch: int, h_kernel_size: int = 11, v_kernel_size: int = 11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x
