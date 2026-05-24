import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class _ASFFBase(nn.Module):
    """Canonical ASFF head-side fusion for three feature levels."""

    def __init__(self, in_channels, level: int, compress_c: int = 8):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)) or len(in_channels) != 3:
            raise ValueError(f"{self.__class__.__name__} expects 3 input channel values, got {in_channels}")

        self.branch_channels = [int(c) for c in in_channels]
        self.level = int(level)
        self.out_channels = self.branch_channels[self.level]
        cc = max(int(compress_c), 1)

        c0, c1, c2 = self.branch_channels

        if self.level == 0:  # target P2
            self.align_level_1 = Conv(c1, self.out_channels, 1, 1)
            self.align_level_2 = Conv(c2, self.out_channels, 1, 1)
        elif self.level == 1:  # target P3
            self.align_level_0 = Conv(c0, self.out_channels, 3, 2)
            self.align_level_2 = Conv(c2, self.out_channels, 1, 1)
        elif self.level == 2:  # target P4
            mid_c = max(c1, self.out_channels // 2)
            self.align_level_0 = nn.Sequential(
                Conv(c0, mid_c, 3, 2),
                Conv(mid_c, self.out_channels, 3, 2),
            )
            self.align_level_1 = Conv(c1, self.out_channels, 3, 2)
        else:
            raise ValueError(f"{self.__class__.__name__} only supports levels 0, 1, 2, got {self.level}")

        self.weight_level_0 = Conv(self.out_channels, cc, 1, 1)
        self.weight_level_1 = Conv(self.out_channels, cc, 1, 1)
        self.weight_level_2 = Conv(self.out_channels, cc, 1, 1)
        self.weight_levels = nn.Conv2d(cc * 3, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.expand = Conv(self.out_channels, self.out_channels, 3, 1)

    @staticmethod
    def _resize(feat, target_hw):
        return F.interpolate(feat, size=target_hw, mode="nearest") if feat.shape[-2:] != target_hw else feat

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise TypeError(f"{self.__class__.__name__} expects a list/tuple with 3 tensors")

        x0, x1, x2 = x

        if self.level == 0:
            target_hw = x0.shape[-2:]
            level_0 = x0
            level_1 = self._resize(self.align_level_1(x1), target_hw)
            level_2 = self._resize(self.align_level_2(x2), target_hw)
        elif self.level == 1:
            target_hw = x1.shape[-2:]
            level_0 = self._resize(self.align_level_0(x0), target_hw)
            level_1 = x1
            level_2 = self._resize(self.align_level_2(x2), target_hw)
        else:
            target_hw = x2.shape[-2:]
            level_0 = self._resize(self.align_level_0(x0), target_hw)
            level_1 = self._resize(self.align_level_1(x1), target_hw)
            level_2 = x2

        weights = torch.cat(
            (
                self.weight_level_0(level_0),
                self.weight_level_1(level_1),
                self.weight_level_2(level_2),
            ),
            dim=1,
        )
        weights = F.softmax(self.weight_levels(weights), dim=1)

        fused = (
            level_0 * weights[:, 0:1]
            + level_1 * weights[:, 1:2]
            + level_2 * weights[:, 2:3]
        )
        return self.expand(fused)


class ASFFP2(_ASFFBase):
    def __init__(self, in_channels, compress_c: int = 8):
        super().__init__(in_channels=in_channels, level=0, compress_c=compress_c)


class ASFFP3(_ASFFBase):
    def __init__(self, in_channels, compress_c: int = 8):
        super().__init__(in_channels=in_channels, level=1, compress_c=compress_c)


class ASFFP4(_ASFFBase):
    def __init__(self, in_channels, compress_c: int = 8):
        super().__init__(in_channels=in_channels, level=2, compress_c=compress_c)
