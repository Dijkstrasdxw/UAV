import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class _AFPNConcatBase(nn.Module):
    """AFPN/ASFF-inspired spatially weighted concat for YOLO neck nodes."""

    def __init__(self, in_channels, n_inputs: int, dimension: int = 1, compress_c: int = 8):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {n_inputs} input channel values, got {in_channels}")
        if int(dimension) != 1:
            raise ValueError(f"{self.__class__.__name__} only supports channel concat (dimension=1), got {dimension}")

        self.branch_channels = [int(c) for c in in_channels]
        self.n_inputs = int(n_inputs)
        self.d = int(dimension)
        cc = max(int(compress_c), 1)

        self.weight_level = nn.ModuleList([Conv(c, cc, 1, 1) for c in self.branch_channels])
        self.weight_levels = nn.Conv2d(cc * self.n_inputs, self.n_inputs, kernel_size=1, stride=1, padding=0, bias=True)

    def _resize_inputs(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple of tensors")
        if len(x) != self.n_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {self.n_inputs} inputs, got {len(x)}")
        target_hw = x[0].shape[-2:]
        return [F.interpolate(xi, size=target_hw, mode="nearest") if xi.shape[-2:] != target_hw else xi for xi in x]

    def _spatial_weights(self, feats):
        weight_feats = torch.cat([head(feat) for head, feat in zip(self.weight_level, feats)], dim=1)
        return F.softmax(self.weight_levels(weight_feats), dim=1)

    def forward(self, x):
        feats = self._resize_inputs(x)
        weights = self._spatial_weights(feats)
        weighted_feats = [feat * weights[:, i : i + 1] for i, feat in enumerate(feats)]
        return torch.cat(weighted_feats, dim=self.d)


class AFPNConcat2(_AFPNConcatBase):
    def __init__(self, in_channels, dimension: int = 1, compress_c: int = 8):
        super().__init__(in_channels=in_channels, n_inputs=2, dimension=dimension, compress_c=compress_c)


class AFPNConcat3(_AFPNConcatBase):
    def __init__(self, in_channels, dimension: int = 1, compress_c: int = 8):
        super().__init__(in_channels=in_channels, n_inputs=3, dimension=dimension, compress_c=compress_c)


class _AFPNWCGBase(_AFPNConcatBase):
    """AFPN spatial weights + WCG-style post-fusion gating."""

    def __init__(
        self,
        in_channels,
        n_inputs: int,
        dimension: int = 1,
        compress_c: int = 8,
        reduction: int = 4,
        gate_kernel: int = 1,
        residual: bool = True,
    ):
        super().__init__(in_channels=in_channels, n_inputs=n_inputs, dimension=dimension, compress_c=compress_c)
        self.out_channels = sum(self.branch_channels)
        self.residual = bool(residual)

        hidden = max(self.out_channels // max(int(reduction), 1), 8)
        self.gate = nn.Sequential(
            Conv(self.out_channels, hidden, k=1, s=1),
            nn.Conv2d(hidden, self.out_channels, kernel_size=gate_kernel, stride=1, padding=gate_kernel // 2, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feats = self._resize_inputs(x)
        weights = self._spatial_weights(feats)

        weighted_feats = [feat * weights[:, i : i + 1] for i, feat in enumerate(feats)]
        fused = torch.cat(weighted_feats, dim=self.d)
        gated = fused * self.gate(fused)
        return fused + gated if self.residual else gated


class AFPNWCGConcat2(_AFPNWCGBase):
    def __init__(
        self,
        in_channels,
        dimension: int = 1,
        compress_c: int = 8,
        reduction: int = 4,
        gate_kernel: int = 1,
        residual: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_inputs=2,
            dimension=dimension,
            compress_c=compress_c,
            reduction=reduction,
            gate_kernel=gate_kernel,
            residual=residual,
        )


class AFPNWCGConcat3(_AFPNWCGBase):
    def __init__(
        self,
        in_channels,
        dimension: int = 1,
        compress_c: int = 8,
        reduction: int = 4,
        gate_kernel: int = 1,
        residual: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            n_inputs=3,
            dimension=dimension,
            compress_c=compress_c,
            reduction=reduction,
            gate_kernel=gate_kernel,
            residual=residual,
        )
