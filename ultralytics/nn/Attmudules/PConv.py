"""PConv operators aligned with FasterNet's Partial_conv3 design."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class PartialConv3(nn.Module):
    """FasterNet Partial_conv3: apply 3x3 conv to only a subset of channels."""

    def __init__(self, dim: int, n_div: int = 4, forward: str = "split_cat"):
        super().__init__()
        if n_div < 1:
            raise ValueError(f"n_div must be >= 1, got {n_div}")

        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        if self.dim_conv3 < 1:
            raise ValueError(f"dim_conv3 becomes 0 (dim={dim}, n_div={n_div})")

        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == "slicing":
            self.forward_impl = self.forward_slicing
        elif forward == "split_cat":
            self.forward_impl = self.forward_split_cat
        else:
            raise NotImplementedError(forward)

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # Only for inference; keep residual branch input unchanged.
        x = x.clone()
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # Training/inference path used in FasterNet.
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(x)


class PConv(nn.Module):
    """YOLO wrapper: PartialConv3 + optional 1x1 projection."""

    def __init__(self, c1: int, c2: int, n_div: int = 4, forward: str = "split_cat", act: bool = True):
        super().__init__()
        self.spatial_mixing = PartialConv3(c1, n_div=n_div, forward=forward)
        self.proj = Conv(c1, c2, 1, 1, act=act) if c1 != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.spatial_mixing(x))
