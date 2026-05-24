import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA(nn.Module):
    """
    ECA (Efficient Channel Attention) Module
    自适应选择卷积核大小 k，用 1D 卷积建模通道间交互
    """

    def __init__(self, channels: int, k_size: int = None, gamma: int = 2, b: int = 1):
        super().__init__()
        assert channels > 0

        # 1) GAP: Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,C,H,W) -> (B,C,1,1)

        # 2) 自适应选择卷积核大小 k
        if k_size is None:
            # 公式: k = floor(abs(log2(C/gamma)) + b)
            t = int(abs(math.log2(channels / gamma)) + b)
            k_size = t if t % 2  else t + 1  # 奇数
        k_size = max(3, k_size)  # 最小为 3

        # 3) 1D 卷积：在通道维度上做局部交互
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,  # 保证输出长度不变
            bias=False
        )

        # 4) Sigmoid: 生成权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: (B, C, H, W)
        b, c, h, w = x.shape

        # --- Step 1: GAP ---
        y = self.gap(x)  # (B, C, 1, 1)

        # --- Step 2: 转换为 1D 张量 ---
        # (B, C, 1, 1) -> (B, C) -> (B, 1, C) → 适配 Conv1d 输入
        y = y.squeeze(-1).squeeze(-1)  # (B, C)
        y = y.unsqueeze(1)           # (B, C) → 等价于 (B, 1, C) for Conv1d
        y = self.conv1d(y)            # (B, 1, C) → (B, 1, C)
        # --- Step 3: Gate ---
        y = self.sigmoid(y)           # (B, 1, C)
        # --- Step 4: Scale ---
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y  # (B, C, H, W)


