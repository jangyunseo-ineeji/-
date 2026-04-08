from __future__ import annotations

import torch
import torch.nn as nn


class ThermalMLP(nn.Module):
    """
    Lightweight prototype: maps concatenated [current field + torch state + boundary]
    to next-step temperatures. Not the full GPyro (no GP head, no physics split).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _col(x: torch.Tensor) -> torch.Tensor:
    """Ensure (batch, 1) for scalar channels."""
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def build_features(
    T_t: torch.Tensor,
    torch_x: torch.Tensor,
    torch_y: torch.Tensor,
    torch_z: torch.Tensor,
    torch_vx: torch.Tensor,
    torch_vy: torch.Tensor,
    torch_flag: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    """Stack inputs along last dim (batch, features)."""
    parts = [
        T_t,
        _col(torch_x),
        _col(torch_y),
        _col(torch_z),
        _col(torch_vx),
        _col(torch_vy),
        _col(torch_flag),
        boundary,
    ]
    return torch.cat(parts, dim=-1)
