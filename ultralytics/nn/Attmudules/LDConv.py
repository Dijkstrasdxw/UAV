import math

import torch
import torch.nn as nn


class LDConv(nn.Module):
    """Linear Deformable Convolution adapted for Ultralytics blocks.

    Notes:
        `num_param` is the number of learned sampling points, not a standard k x k kernel size.
    """

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, num_param: int = 3, stride: int = 1, bias: bool = False, act=True):
        super().__init__()
        if num_param < 1:
            raise ValueError(f"num_param must be >= 1, got {num_param}")

        self.num_param = int(num_param)
        self.stride = int(stride)

        self.proj = nn.Conv2d(
            c1,
            c2,
            kernel_size=(self.num_param, 1),
            stride=(self.num_param, 1),
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.offset = nn.Conv2d(c1, 2 * self.num_param, kernel_size=3, stride=self.stride, padding=1, bias=True)
        nn.init.constant_(self.offset.weight, 0.0)
        nn.init.constant_(self.offset.bias, 0.0)

        base_coords = self._build_base_coords(self.num_param, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("base_coords", base_coords, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        n = self.num_param
        p = self._get_sampling_locations(offset)

        # (b, 2N, h, w) -> (b, h, w, 2N)
        p = p.permute(0, 2, 3, 1).contiguous()

        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :n], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., n:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :n], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., n:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :n], q_rb[..., n:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :n], q_lt[..., n:]], dim=-1)

        p = torch.cat(
            [
                torch.clamp(p[..., :n], 0, x.size(2) - 1),
                torch.clamp(p[..., n:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        g_lt = (1 + (q_lt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_lt[..., n:].type_as(p) - p[..., n:]))
        g_rb = (1 - (q_rb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_rb[..., n:].type_as(p) - p[..., n:]))
        g_lb = (1 + (q_lb[..., :n].type_as(p) - p[..., :n])) * (1 - (q_lb[..., n:].type_as(p) - p[..., n:]))
        g_rt = (1 - (q_rt[..., :n].type_as(p) - p[..., :n])) * (1 + (q_rt[..., n:].type_as(p) - p[..., n:]))

        x_q_lt = self._sample_from_index(x, q_lt)
        x_q_rb = self._sample_from_index(x, q_rb)
        x_q_lb = self._sample_from_index(x, q_lb)
        x_q_rt = self._sample_from_index(x, q_rt)

        x_offset = (
            g_lt.unsqueeze(1) * x_q_lt
            + g_rb.unsqueeze(1) * x_q_rb
            + g_lb.unsqueeze(1) * x_q_lb
            + g_rt.unsqueeze(1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset)

        y = self.proj(x_offset)
        return self.act(self.bn(y))

    @staticmethod
    def _build_base_coords(num_param: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = max(round(math.sqrt(num_param)), 1)
        rows = num_param // base
        remainder = num_param % base

        x_main, y_main = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(base, device=device),
            indexing="ij",
        )
        x_coords = x_main.reshape(-1)
        y_coords = y_main.reshape(-1)

        if remainder > 0:
            x_tail, y_tail = torch.meshgrid(
                torch.arange(rows, rows + 1, device=device),
                torch.arange(remainder, device=device),
                indexing="ij",
            )
            x_coords = torch.cat((x_coords, x_tail.reshape(-1)), dim=0)
            y_coords = torch.cat((y_coords, y_tail.reshape(-1)), dim=0)

        coords = torch.cat((x_coords, y_coords), dim=0).view(1, 2 * num_param, 1, 1)
        return coords.to(dtype=dtype)

    def _get_p_0(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x_base, y_base = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride, device=device),
            torch.arange(0, w * self.stride, self.stride, device=device),
            indexing="ij",
        )
        x_base = x_base.reshape(1, 1, h, w).repeat(1, self.num_param, 1, 1)
        y_base = y_base.reshape(1, 1, h, w).repeat(1, self.num_param, 1, 1)
        return torch.cat((x_base, y_base), dim=1).to(dtype=dtype)

    def _get_sampling_locations(self, offset: torch.Tensor) -> torch.Tensor:
        _, _, h, w = offset.shape
        p_n = self.base_coords.to(device=offset.device, dtype=offset.dtype)
        p_0 = self._get_p_0(h, w, offset.device, offset.dtype)
        return p_0 + p_n + offset

    def _sample_from_index(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = q.shape
        c = x.size(1)
        width = x.size(3)

        x_flat = x.contiguous().view(b, c, -1)
        index = q[..., : self.num_param] * width + q[..., self.num_param :]
        index = index.unsqueeze(1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        return x_flat.gather(dim=-1, index=index).contiguous().view(b, c, h, w, self.num_param)

    def _reshape_x_offset(self, x_offset: torch.Tensor) -> torch.Tensor:
        b, c, h, w, _ = x_offset.shape
        return x_offset.permute(0, 1, 2, 4, 3).contiguous().view(b, c, h * self.num_param, w)
