from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gpyro_prototype.model import build_features


def exclude_cross_segment_timesteps(indices: np.ndarray, segment_starts: np.ndarray) -> np.ndarray:
    """
    Drop timestep indices t where t and t+1 belong to different concatenated experiments.
    segment_starts: cumulative start positions, length n_experiments + 1.
    """
    if segment_starts is None or len(segment_starts) <= 2:
        return indices
    invalid = segment_starts[1:-1] - 1
    mask = ~np.isin(indices, invalid)
    return indices[mask]


@dataclass
class ContiguousSplit:
    train: slice
    val: slice
    test: slice
    n_total: int


def contiguous_split_indices(n: int, train_frac: float, val_frac: float) -> ContiguousSplit:
    """
    Time-ordered contiguous segments (no shuffle).
    For one-step targets T[t+1], each region only uses t where both t and t+1
    stay inside the same region (no cross-boundary supervision leakage).
    """
    if n < 8:
        raise ValueError("Need a longer series for train/val/test splits.")
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_train = max(3, min(n_train, n - 5))
    n_val = max(3, min(n_val, n - n_train - 3))
    n_test = n - n_train - n_val
    if n_test < 3:
        n_val = max(3, n_val - 1)
        n_test = n - n_train - n_val

    # t ranges (inclusive lower, exclusive upper for arange): need t+1 < region_end
    tr_end = n_train - 1  # t in [0, n_train-2]
    va_end = n_train + n_val - 1  # t in [n_train, n_train+n_val-2]
    te_end = n - 1  # t in [n_train+n_val, n-2]
    return ContiguousSplit(
        train=slice(0, tr_end),
        val=slice(n_train, va_end),
        test=slice(n_train + n_val, te_end),
        n_total=n,
    )


class ThermalSequenceDataset(Dataset):
    """One-step-ahead prediction: x_t -> T_{t+1}."""

    def __init__(self, bundle: dict[str, Any], index_slice: slice, device: torch.device | None = None):
        self.bundle = bundle
        self.sl = index_slice
        self.device = device
        T = bundle["T"]
        self.start = index_slice.start or 0
        self.stop = index_slice.stop if index_slice.stop is not None else len(T) - 1
        # Need t and t+1 inside [start, stop)
        self.stop = min(self.stop, len(T) - 1)
        raw = np.arange(self.start, self.stop, dtype=np.int64)
        ss = bundle.get("segment_starts")
        if ss is not None:
            raw = exclude_cross_segment_timesteps(raw, ss)
        self.indices = raw

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        T = self.bundle["T"]
        feat = build_features(
            torch.from_numpy(T[t : t + 1]),
            torch.from_numpy(self.bundle["torch_x"][t : t + 1]),
            torch.from_numpy(self.bundle["torch_y"][t : t + 1]),
            torch.from_numpy(self.bundle["torch_z"][t : t + 1]),
            torch.from_numpy(self.bundle["torch_vx"][t : t + 1]),
            torch.from_numpy(self.bundle["torch_vy"][t : t + 1]),
            torch.from_numpy(self.bundle["torch_flag"][t : t + 1]),
            torch.from_numpy(self.bundle["boundary"][t : t + 1]),
        ).squeeze(0)
        y = torch.from_numpy(T[t + 1])
        if self.device is not None:
            feat = feat.to(self.device)
            y = y.to(self.device)
        return feat, y


def bundle_to_numpy(bundle: dict[str, Any]) -> dict[str, Any]:
    """Ensure arrays are numpy float32 for fast slicing."""
    out = dict(bundle)
    for k in ["T", "torch_x", "torch_y", "torch_z", "torch_vx", "torch_vy", "torch_flag", "boundary", "time_s"]:
        if k in out:
            out[k] = np.asarray(bundle[k], dtype=np.float32)
    if "segment_starts" in out:
        out["segment_starts"] = np.asarray(bundle["segment_starts"], dtype=np.int64)
    return out
