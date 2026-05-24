import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """
    Squeeze-and-Excitation (SE) 模块
    作用：学习通道间的依赖关系，对每个通道赋予不同的权重
    输入：(B, C, H, W)
    输出：(B, C, H, W)，但每个通道被重新加权
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        初始化 SE 模块

        Args:
            channels: 输入特征图的通道数 C
            reduction: 压缩比，hidden_dim = C // reduction（常用 16）
        """
        super().__init__()
        assert channels > 0, "通道数必须大于0"

        # 1. Squeeze: 全局平均池化 (B,C,H,W) -> (B,C,1,1)
        # 将空间维度 H×W 压缩成 1×1，保留通道信息
        self.gap = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化到 1x1

        # 2. Excitation: 两层"通道MLP"
        # (B,C,1,1) -> (B, hidden, 1,1) -> (B,C,1,1)
        hidden = max(1, channels // reduction)  # 防止C太小时hidden=0
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)  # 第一层FC
        self.act = nn.ReLU(inplace=True)  # 激活函数
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)  # 第二层FC

        # 3. Scale: sigmoid 压缩到 [0,1] 范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            out: 加权后的输出 (B, C, H, W)
        """
        b, c, h, w = x.shape  # 仅用于理解维度

        # ---- Squeeze: 全局平均池化 ----
        # (B,C,H,W) -> (B,C,1,1)
        z = self.gap(x)  # z 是每个通道的全局平均值
        # ---- Excitation: 两层MLP学习通道权重 ----
        # (B,C,1,1) -> (B,hidden,1,1) -> (B,C,1,1)
        s = self.fc2(self.act(self.fc1(z)))  # 两层卷积 + ReLU
        s = self.sigmoid(s)  # 压缩到 [0,1] 区间，作为权重

        # ---- Scale: 广播乘法 ----
        # s: (B,C,1,1)，在 H,W 维度上广播，所以输出仍为 (B,C,H,W)
        out = x * s  # 每个通道乘以对应权重

        return out
