import torch
import torch.nn as nn
import torch.nn.functional as F


class _BiFPNAddBase(nn.Module):
    """BiFPN-style weighted sum with optional channel alignment for YOLO feature fusion."""

    def __init__(self, in_channels, n_inputs: int, out_channels: int | None = None, eps: float = 1e-4):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {n_inputs} input channel values, got {in_channels}")

        self.in_channels = [int(c) for c in in_channels]
        self.n_inputs = int(n_inputs)
        self.out_channels = int(out_channels) if out_channels is not None else self.in_channels[0]
        self.eps = float(eps)
        self.w = nn.Parameter(torch.ones(self.n_inputs, dtype=torch.float32), requires_grad=True)
        self.align = nn.ModuleList(
            nn.Identity() if c == self.out_channels else nn.Conv2d(c, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            for c in self.in_channels
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple of tensors")
        if len(x) != self.n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {self.n_inputs} inputs, got {len(x)}")

        target_hw = x[0].shape[-2:]
        feats = []
        for feat, align in zip(x, self.align):
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="nearest")
            feats.append(align(feat))

        w = F.relu(self.w, inplace=False)
        weight = w / (torch.sum(w, dim=0) + self.eps)
        return sum(weight[i] * feats[i] for i in range(self.n_inputs))


class BiFPNAdd2(_BiFPNAddBase):
    def __init__(self, in_channels, out_channels: int | None = None, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=2, out_channels=out_channels, eps=eps)


class BiFPNAdd3(_BiFPNAddBase):
    def __init__(self, in_channels, out_channels: int | None = None, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=3, out_channels=out_channels, eps=eps)


class _BiFPNConcatBase(nn.Module):
    """YOLO-adapted BiFPN fusion: weighted concat followed by lightweight post-fusion refinement."""

    def __init__(self, in_channels: int, n_inputs: int, dimension: int = 1, eps: float = 1e-4):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_inputs = int(n_inputs)
        self.d = int(dimension)
        self.eps = float(eps)
        self.w = nn.Parameter(torch.ones(self.n_inputs, dtype=torch.float32), requires_grad=True)
        self.act = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True)

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
        fused = torch.cat([weight[i] * feats[i] for i in range(self.n_inputs)], dim=self.d)

        if fused.shape[1] != self.in_channels:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.in_channels} fused channels, got {fused.shape[1]}."
            )

        return self.conv(self.act(fused))


class BiFPNConcat2(_BiFPNConcatBase):
    def __init__(self, in_channels: int, dimension: int = 1, eps: float = 1e-4):
        super().__init__(in_channels=in_channels, n_inputs=2, dimension=dimension, eps=eps)


class BiFPNConcat3(_BiFPNConcatBase):
    def __init__(self, in_channels: int, dimension: int = 1, eps: float = 1e-4):
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
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, groups=self.out_channels, bias=False),
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
