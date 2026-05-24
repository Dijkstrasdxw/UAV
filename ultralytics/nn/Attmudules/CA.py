import torch
import torch.nn as nn


class CA(nn.Module):
    """
    Coordinate Attention Module (CA)
    分别提取高度和宽度方向的坐标信息，生成空间注意力图
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # 确保通道数大于0
        assert channels > 0

        # 计算中间通道数（压缩比）
        mip = max(8, channels // reduction)  # 论文里常设最小8

        # 共享变换：C -> mip （1x1 conv）
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        # 两个分支：mip -> C
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # --- Step 1: 分方向池化（保留坐标信息）---
        # x_h: (B, C, H, 1) → 沿 W 平均（保留 H 信息）
        x_h = x.mean(dim=3, keepdim=True)  # dim=3 是 W

        # x_w: (B, C, 1, W) → 沿 H 平均（保留 W 信息）
        x_w = x.mean(dim=2, keepdim=True)  # dim=2 是 H

        # --- Step 2: 拼接（在空间维度拼起来）---
        # 为了拼接，把 x_w 转换成 (B, C, W, 1)，这样和 x_h 的最后一维都是 1
        x_w_t = x_w.permute(0, 1, 3, 2)  # (B, C, W, 1)
        y = torch.cat([x_h, x_w_t], dim=2)  # (B, C, H+W, 1)

        # --- Step 3: 共享变换（瓶颈）---
        y = self.act(self.bn1(self.conv1(y)))  # (B, mip, H+W, 1)

        # --- Step 4: 拆分两支 ---
        y_h, y_w = torch.split(y, split_size_or_sections=[h, w], dim=2)  # (B, mip, H, 1), (B, mip, W, 1)

        # 把 y_w 还原回 (B, mip, 1, W)
        y_w = y_w.permute(0, 1, 3, 2)  # (B, mip, 1, W)

        # --- Step 5: 生成两个方向的注意力图 ---
        a_h = self.sigmoid(self.conv_h(y_h))  # (B, C, H, 1)
        a_w = self.sigmoid(self.conv_w(y_w))  # (B, C, 1, W)

        # --- Step 6: 重标定（广播逐元素相乘）---
        # a_h 会广播到 W，a_w 会广播到 H
        out = x * a_h * a_w  # (B, C, H, W)

        return out