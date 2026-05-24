import torch
import torch.nn as nn
import torch.nn.functional as F


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Bi_FPN(nn.Module):
    """Blog-style weighted-add BiFPN fusion."""

    def __init__(self, length: int, eps: float = 1e-4):
        super().__init__()
        self.length = int(length)
        self.epsilon = float(eps)
        self.weight = nn.Parameter(torch.ones(self.length, dtype=torch.float32), requires_grad=True)
        self.swish = swish()

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple of tensors")
        if len(x) != self.length:
            raise ValueError(f"{self.__class__.__name__} expects {self.length} inputs, got {len(x)}")

        target_hw = x[0].shape[-2:]
        feats = [F.interpolate(xi, size=target_hw, mode="nearest") if xi.shape[-2:] != target_hw else xi for xi in x]

        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
        weighted_feature_maps = [weights[i] * feats[i] for i in range(len(feats))]
        stacked_feature_maps = torch.stack(weighted_feature_maps, dim=0)
        return torch.sum(stacked_feature_maps, dim=0)


class _BiFPNChannelConcatBase(nn.Module):
    """BiFPN-style concat with static channel-level reweighting inspired by FFCA."""

    def __init__(self, in_channels, n_inputs: int, dimension: int = 1, eps: float = 1e-4):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {n_inputs} input channel values, got {in_channels}")

        self.branch_channels = [int(c) for c in in_channels]
        self.n_inputs = int(n_inputs)
        self.d = int(dimension)
        self.eps = float(eps)
        self.out_channels = sum(self.branch_channels)
        self.w = nn.Parameter(torch.ones(self.out_channels, dtype=torch.float32), requires_grad=True)

        if self.d != 1:
            raise ValueError(f"{self.__class__.__name__} only supports channel concat (dimension=1), got {self.d}")

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple of tensors")
        if len(x) != self.n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {self.n_inputs} inputs, got {len(x)}")

        target_hw = x[0].shape[-2:]
        feats = [F.interpolate(xi, size=target_hw, mode="nearest") if xi.shape[-2:] != target_hw else xi for xi in x]

        w = F.relu(self.w, inplace=False)
        weight = w / (torch.sum(w, dim=0) + self.eps)

        weighted_feats = []
        start = 0
        for feat, c in zip(feats, self.branch_channels):
            end = start + c
            weighted_feats.append(feat * weight[start:end].view(1, c, 1, 1))
            start = end
        return torch.cat(weighted_feats, dim=self.d)


class BiFPNConcat2(_BiFPNChannelConcatBase):
    def __init__(self, in_channels, dimension: int = 1, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=2, dimension=dimension, eps=eps)


class BiFPNConcat3(_BiFPNChannelConcatBase):
    def __init__(self, in_channels, dimension: int = 1, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=3, dimension=dimension, eps=eps)


class _BiFPNConcatDynBase(nn.Module):
    """BiFPN-inspired weighted concat with dynamic branch scoring and lightweight spatial refinement."""

    def __init__(self, in_channels, n_inputs: int, reduction: int = 4, eps: float = 1e-4):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {n_inputs} input channel values, got {in_channels}")

        self.branch_channels = [int(c) for c in in_channels]
        self.n_inputs = int(n_inputs)
        self.out_channels = sum(self.branch_channels)
        self.eps = float(eps)
        self.reduction = max(int(reduction), 1)
        self.w = nn.Parameter(torch.ones(self.n_inputs, dtype=torch.float32), requires_grad=True)
        self.alpha = nn.Parameter(torch.full((self.n_inputs,), 0.5, dtype=torch.float32), requires_grad=True)

        self.branch_score = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(2 * c, max(c // self.reduction, 8), kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(c // self.reduction, 8), 1, kernel_size=1, bias=True),
            )
            for c in self.branch_channels
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.SiLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple of tensors")
        if len(x) != self.n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {self.n_inputs} inputs, got {len(x)}")

        target_hw = x[0].shape[-2:]
        feats = [F.interpolate(xi, size=target_hw, mode="nearest") if xi.shape[-2:] != target_hw else xi for xi in x]

        static_w = F.relu(self.w, inplace=False)
        static_w = static_w / (torch.sum(static_w, dim=0) + self.eps)

        dyn_logits = []
        for feat, score_block in zip(feats, self.branch_score):
            pooled = torch.cat((self.avg_pool(feat), self.max_pool(feat)), dim=1)
            dyn_logits.append(score_block(pooled))
        dyn_logits = torch.cat(dyn_logits, dim=1)
        dyn_w = F.softmax(dyn_logits, dim=1)

        alpha = torch.sigmoid(self.alpha).view(1, self.n_inputs, 1, 1)
        static_w = static_w.view(1, self.n_inputs, 1, 1)
        mix_w = alpha * static_w + (1.0 - alpha) * dyn_w
        mix_w = mix_w / (mix_w.sum(dim=1, keepdim=True) + self.eps)

        fused = torch.cat([feat * mix_w[:, i : i + 1] for i, feat in enumerate(feats)], dim=1)
        return self.refine(fused)


class BiFPNConcatDyn2(_BiFPNConcatDynBase):
    def __init__(self, in_channels, reduction: int = 4, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=2, reduction=reduction, eps=eps)


class BiFPNConcatDyn3(_BiFPNConcatDynBase):
    def __init__(self, in_channels, reduction: int = 4, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=3, reduction=reduction, eps=eps)
