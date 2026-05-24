import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力，提升特征表示能力
    输入：(B, C, H, W)
    输出：(B, C, H, W)，但每个通道和位置被重新加权
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        初始化 CBAM 模块

        Args:
            channels: 输入特征图的通道数 C
            reduction: 通道注意力中 MLP 的压缩比（默认 16）
            kernel_size: 空间注意力中卷积核大小（默认 7）
        """
        super(CBAM, self).__init__()
        assert channels > 0, "通道数必须大于0"
        hidden = max(channels // reduction, 1)

        # 1. Channel Attention: 通道注意力模块
        # 使用 AvgPool 和 MaxPool 提取全局信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B,C,H,W) -> (B,C,1,1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # (B,C,H,W) -> (B,C,1,1)

        # 共享的 MLP（两层全连接层），用于学习通道间依赖
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        )

        # Sigmoid 压缩到 [0,1] 范围
        self.sigmoid = nn.Sigmoid()

        # 2. Spatial Attention: 空间注意力模块
        # 用 7x7 卷积融合通道信息
        self.spatial_conv = nn.Conv2d(in_channels=2, out_channels=1,
                                      kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 (B, C, H, W)

        Returns:
            out: 加权后的输出 (B, C, H, W)
        """
        b, c, h, w = x.shape

        # ---- 1. Channel Attention ----
        # (1) 全局平均池化和最大池化
        avg = self.avg_pool(x)  # (B,C,1,1)
        mx = self.max_pool(x)   # (B,C,1,1)

        # (2) 共享 MLP 处理
        # 将 (B,C,1,1) → (B,C,1,1)（通过 MLP 学习通道关系）
        mc = self.sigmoid(self.mlp(avg) + self.mlp(mx))

        # (3) 通道加权（广播）
        x_c = x * mc  # (B,C,H,W) ← 重要通道被加强

        # ---- 2. Spatial Attention ----
        # (1) 对通道维度做 mean/max 操作
        # avg_c: (B,1,H,W) - 每个位置的平均通道值
        # max_c: (B,1,H,W) - 每个位置的最大通道值
        avg_c = x_c.mean(dim=1, keepdim=True)  # (B,1,H,W)
        max_c = x_c.max(dim=1, keepdim=True)[0]  # (B,1,H,W)

        # (2) concat 后用 7x7 卷积融合
        concat = torch.cat([avg_c, max_c], dim=1)  # (B,2,H,W)
        ms = self.spatial_sigmoid(self.spatial_conv(concat))  # (B,1,H,W)

        # (3) 空间加权（广播）
        out = x_c * ms  # (B,C,H,W) ← 重要区域被加强

        return out
