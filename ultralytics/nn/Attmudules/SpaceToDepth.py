import torch
import torch.nn as nn


class SpaceToDepth(nn.Module):
    """Space-to-depth downsampling layer (aligned with SPD-Conv official ordering)."""

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = int(scale)
        if self.scale != 2:
            raise ValueError(f"SpaceToDepth currently matches official SPD-Conv for scale=2 only, got scale={self.scale}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official SPD-Conv ordering: [00, 10, 01, 11]
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
