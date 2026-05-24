import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class PKIModule(nn.Module):
    """YOLO-adapted Poly Kernel Inception block with shape-preserving multi-kernel DWConv branches."""

    def __init__(
        self,
        c1: int,
        c2: int,
        e: float = 1.0,
        kernel_sizes=(3, 5, 7, 9, 11),
        shortcut: bool = True,
    ):
        super().__init__()
        kernels = tuple(int(k) for k in kernel_sizes)
        if len(kernels) < 2:
            raise ValueError(f"PKIModule expects at least 2 kernel sizes, got {kernels}")
        if any(k % 2 == 0 or k < 3 for k in kernels):
            raise ValueError(f"PKIModule expects odd kernel sizes >= 3, got {kernels}")

        hidden = max(int(c2 * e), 1)
        self.pre = Conv(c1, hidden, k=1, s=1)

        # Keep the official PKI spirit: use a dense 3x3 DWConv as the base feature,
        # then aggregate larger depthwise kernels in parallel instead of using dilation.
        self.base = Conv(hidden, hidden, k=kernels[0], s=1, g=hidden, act=False)
        self.branches = nn.ModuleList([Conv(hidden, hidden, k=k, s=1, g=hidden, act=False) for k in kernels[1:]])
        self.fuse = Conv(hidden, c2, k=1, s=1)
        self.shortcut = shortcut and c1 == c2
        self.kernel_sizes = kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        base = self.base(y)
        y = base
        for branch in self.branches:
            y = y + branch(base)
        y = self.fuse(y)
        return x + y if self.shortcut else y
