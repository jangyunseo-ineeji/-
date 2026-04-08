from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXPERIMENT_IDS: tuple[str, ...] = tuple(f"T{i}" for i in range(1, 11))  # T1..T10


@dataclass(frozen=True)
class PrototypeConfig:
    """Tunable knobs for the development prototype."""

    data_root: Path
    # Load & concatenate these experiments (time-ordered concat). Default: first 10 (T1..T10).
    experiment_ids: tuple[str, ...] = DEFAULT_EXPERIMENT_IDS

    # Contiguous splits along time (fractions of sequence length)
    train_frac: float = 0.70
    val_frac: float = 0.15
    # test_frac = remainder

    # Autoregressive rollout length for paper-style metrics (keep small for dev speed)
    rollout_horizon: int = 48

    # DTW is O(n*m); subsample aligned series by this stride before DTW (dev speed)
    dtw_subsample_stride: int = 8

    # Avoid division by tiny |T_meas| in relative errors
    relative_eps_c: float = 1.0

    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 15
    hidden_dim: int = 256
    seed: int = 42

    @property
    def recorded_points_path(self) -> Path:
        return self.data_root / "Recorded_points.csv"

    def temperatures_path(self, experiment_id: str) -> Path:
        return self.data_root / "temperatures" / f"{experiment_id}_corrected.csv"

    def coordinates_path(self, experiment_id: str) -> Path:
        return self.data_root / "Coordinate_Time" / f"Coordinates_{experiment_id}.csv"

    def raw_path(self, experiment_id: str) -> Path:
        return self.data_root / "raw" / f"{experiment_id}.csv"


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "GPyro-TD"
