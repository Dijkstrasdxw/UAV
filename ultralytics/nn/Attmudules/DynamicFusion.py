import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class _DynamicConv(nn.Module):
    """Standard dynamic 3x3 conv with input-dependent expert mixing."""

    def __init__(self, channels: int, num_experts: int = 4, reduction: int = 4):
        super().__init__()
        channels = int(channels)
        num_experts = max(int(num_experts), 1)
        hidden = max(channels // max(int(reduction), 1), 16)

        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_experts, kernel_size=1, bias=True),
        )
        self.experts = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
            for _ in range(num_experts)
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        alpha = F.softmax(self.router(x).flatten(1), dim=1)
        y = 0
        for i, expert in enumerate(self.experts):
            y = y + expert(x) * alpha[:, i].view(-1, 1, 1, 1)
        return self.act(self.bn(y))


class _WeightedDynamicFusionBase(nn.Module):
    """Weighted neck fusion: branch weights -> concat -> dynamic 3x3 -> 1x1 channel restore."""

    def __init__(self, in_channels, n_inputs: int, out_channels: int, num_experts: int = 4, reduction: int = 4):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {n_inputs} input channel values, got {in_channels}")

        self.in_channels = [int(c) for c in in_channels]
        self.n_inputs = int(n_inputs)
        self.concat_channels = sum(self.in_channels)
        self.out_channels = int(out_channels)

        self.branch_score = nn.ModuleList(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, max(c // 4, 8), kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(c // 4, 8), 1, kernel_size=1, bias=True),
            )
            for c in self.in_channels
        )
        self.dynamic_conv = _DynamicConv(
            channels=self.concat_channels,
            num_experts=num_experts,
            reduction=reduction,
        )
        self.channel_restore = Conv(self.concat_channels, self.out_channels, k=1, s=1)

    def _resize_inputs(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple of tensors")
        if len(x) != self.n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {self.n_inputs} inputs, got {len(x)}")
        target_hw = x[0].shape[-2:]
        return [F.interpolate(xi, size=target_hw, mode="nearest") if xi.shape[-2:] != target_hw else xi for xi in x]

    def forward(self, x):
        feats = self._resize_inputs(x)
        logits = torch.cat([score(feat) for score, feat in zip(self.branch_score, feats)], dim=1)
        weights = F.softmax(logits, dim=1)
        fused = torch.cat([feat * weights[:, i : i + 1] for i, feat in enumerate(feats)], dim=1)
        fused = self.dynamic_conv(fused)
        return self.channel_restore(fused)


class DynamicFusion2(_WeightedDynamicFusionBase):
    def __init__(self, in_channels, out_channels: int, num_experts: int = 4, reduction: int = 4):
        super().__init__(
            in_channels=in_channels,
            n_inputs=2,
            out_channels=out_channels,
            num_experts=num_experts,
            reduction=reduction,
        )


class DynamicFusion3(_WeightedDynamicFusionBase):
    def __init__(self, in_channels, out_channels: int, num_experts: int = 4, reduction: int = 4):
        super().__init__(
            in_channels=in_channels,
            n_inputs=3,
            out_channels=out_channels,
            num_experts=num_experts,
            reduction=reduction,
        )
