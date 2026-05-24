import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from .C2f_PConv import FasterBlockPConv


class C3k2_PConv(nn.Module):
    """C3k2-style CSP block with PConv inner units.

    This keeps the external C3k2/C2f-style shell and only replaces the repeated
    inner units with FasterNet-style PConv blocks.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        pconv_forward: str = "split_cat",
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        inner_depth = 2 if c3k else 1
        self.m = nn.ModuleList(
            nn.Sequential(
                *(
                    FasterBlockPConv(
                        self.c,
                        shortcut=shortcut,
                        n_div=n_div,
                        mlp_ratio=mlp_ratio,
                        pconv_forward=pconv_forward,
                    )
                    for _ in range(inner_depth)
                )
            )
            for _ in range(n)
        )
        _ = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
