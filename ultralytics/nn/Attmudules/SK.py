import torch
import torch.nn as nn
import torch.nn.functional as F


class SK(nn.Module):
    """
    SK (Selective Kernel) Attention Module
    多分支卷积 + 动态权重选择，增强对不同感受野的适应能力
    """

    def __init__(
            self,channels: int, kernels: tuple = (3, 5), reduction: int = 16,L: int = 32 ,groups: int = 1, act: str = "silu"):
        super().__init__()

        # C : 输入通道数
        self.channels = channels
        # kernels : 分支卷积核大小（常用 3/5 或 3/7）
        self.kernels = list(kernels)
        # M : 分支数
        self.M = len(self.kernels)
        # reduction : 压缩比
        self.reduction = reduction
        # L : bottleneck 最小维度（论文中常用）
        self.L = L
        # groups : 分组卷积（通常设为 1，但可设为 channels // depthwise）
        self.groups = groups
        # act : 激活函数（默认 silu）
        self.act = act

        # 激活函数设置
        if act.lower() == "silu":
            self.act_act = nn.SiLU(inplace=True)
        elif act.lower() == "relu":
            self.act_act = nn.ReLU(inplace=True)
        else:
            self.act_act = nn.SiLU(inplace=True)

        # --- 1) Split: 多分支卷积 ---
        # 每个分支输出 shape 是 (B, C, H, W)
        self.branches = nn.ModuleList()
        for k in self.kernels:
            p = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, kernel_size=k, padding=p, groups=self.groups, bias=False),
                    nn.BatchNorm2d(self.channels),
                    self.act_act
                )
            )

        # --- 2) Fuse: 融合后 GAP ---
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B,C,H,W) -> (B,C,1,1)

        # --- 3) Select: 生成分支权重 ---
        # (B,C,1,1) -> (B,d,1,1)
        d = max(self.channels // self.reduction, self.L)
        self.fc = nn.Sequential(
            nn.Conv2d(self.channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            self.act_act
        )
        # 为每个分支生成 (B,C,1,1) 的 logits
        self.fc_branches = nn.ModuleList([nn.Conv2d(d, self.channels, kernel_size=1, bias=True) for _ in range(self.M)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Split ---
        # feats: list, 每个元素 (B, C, H, W)
        feats = [br(x) for br in self.branches]
        # feats_stack: (B, M, C, H, W)
        feats_stack = torch.stack(feats, dim=1)

        # --- Fuse ---
        # U: (B, C, H, W) （各分支求和融合得到全局特征）
        U = feats_stack.sum(dim=1)
        # s: (B, C, 1, 1)
        s = self.gap(U)
        # z: (B, d, 1, 1)
        z = self.fc(s)

        # --- Select ---
        # logits: (B, M, C, 1, 1) 每个分支一份通道权重logits
        logits = torch.stack([fc(z) for fc in self.fc_branches], dim=1)
        # attn: (B, M, C,1, 1) 在分支维上 softmax，得到各分支权重
        attn = F.softmax(logits, dim=1)

        # --- Weighted Sum ---
        # 对每个分支乘以对应权重，再相加
        out = (feats_stack *attn).sum(dim=1)

        return out

