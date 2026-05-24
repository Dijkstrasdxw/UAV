import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class _CondRouting(nn.Module):
    """CondConv routing function: global pooling + linear projection + sigmoid."""

    def __init__(self, in_channels: int, num_experts: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(in_channels), int(num_experts))

    def forward(self, x):
        x = self.avgpool(x).flatten(1)
        return torch.sigmoid(self.fc(x))


class _CondConv2d(nn.Module):
    """CondConv-style 3x3 convolution with per-sample kernel aggregation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, num_experts: int = 2, bias: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.num_experts = int(num_experts)
        self.padding = self.kernel_size // 2

        self.routing = _CondRouting(self.in_channels, self.num_experts)
        self.weight = nn.Parameter(
            torch.randn(self.num_experts, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_channels)) if bias else None
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.SiLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i], a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        b, _, h, w = x.shape
        routing_weights = self.routing(x)  # [B, K]

        weight = self.weight.view(self.num_experts, -1)
        aggregate_weight = torch.matmul(routing_weights, weight).view(
            b * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        aggregate_bias = None
        if self.bias is not None:
            aggregate_bias = torch.matmul(routing_weights, self.bias).view(-1)

        x = x.view(1, b * self.in_channels, h, w)
        y = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=1, padding=self.padding, groups=b)
        y = y.view(b, self.out_channels, y.shape[-2], y.shape[-1])
        return self.act(self.bn(y))


class _CondFusionBase(nn.Module):
    """Weighted concat -> CondConv 3x3 -> 1x1 channel restore."""

    def __init__(self, in_channels, n_inputs: int, out_channels: int, num_experts: int = 2):
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
        self.cond_conv = _CondConv2d(
            in_channels=self.concat_channels,
            out_channels=self.concat_channels,
            kernel_size=3,
            num_experts=num_experts,
            bias=False,
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
        fused = self.cond_conv(fused)
        return self.channel_restore(fused)


class CondFusion2(_CondFusionBase):
    def __init__(self, in_channels, out_channels: int, num_experts: int = 2):
        super().__init__(in_channels=in_channels, n_inputs=2, out_channels=out_channels, num_experts=num_experts)


class CondFusion3(_CondFusionBase):
    def __init__(self, in_channels, out_channels: int, num_experts: int = 2):
        super().__init__(in_channels=in_channels, n_inputs=3, out_channels=out_channels, num_experts=num_experts)
