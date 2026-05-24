import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class WeightedConcatGate(nn.Module):
    """Lightweight concat replacement with branch-adaptive weights and post-fusion gating."""

    def __init__(self, in_channels, reduction: int = 4, gate_kernel: int = 1, residual: bool = True):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) < 2:
            raise ValueError(f"WeightedConcatGate expects a channel list with at least 2 inputs, got {in_channels}")

        self.in_channels = [int(c) for c in in_channels]
        self.n = len(self.in_channels)
        self.out_channels = sum(self.in_channels)
        self.residual = residual

        hidden = max(self.out_channels // max(int(reduction), 1), 8)
        self.branch_score = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, max(c // 4, 8), kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(c // 4, 8), 1, kernel_size=1, bias=True),
            )
            for c in self.in_channels
        )

        # Gate the fused tensor without changing the channel count.
        self.gate = nn.Sequential(
            Conv(self.out_channels, hidden, k=1, s=1),
            nn.Conv2d(hidden, self.out_channels, kernel_size=gate_kernel, stride=1, padding=gate_kernel // 2, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError("WeightedConcatGate expects a list/tuple of tensors")
        if len(x) != self.n:
            raise ValueError(f"WeightedConcatGate expects {self.n} inputs, got {len(x)}")

        target_hw = x[0].shape[-2:]
        feats = [F.interpolate(xi, size=target_hw, mode='nearest') if xi.shape[-2:] != target_hw else xi for xi in x]

        logits = torch.cat([score(feat) for score, feat in zip(self.branch_score, feats)], dim=1)
        weights = F.softmax(logits, dim=1)

        weighted = [feat * weights[:, i : i + 1] for i, feat in enumerate(feats)]
        fused = torch.cat(weighted, dim=1)
        gated = fused * self.gate(fused)
        return fused + gated if self.residual else gated
