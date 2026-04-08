from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow running as script: python gpyro_prototype/train.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpyro_prototype.config import DEFAULT_DATA_ROOT, DEFAULT_EXPERIMENT_IDS, PrototypeConfig
from gpyro_prototype.dataset import ContiguousSplit, ThermalSequenceDataset, bundle_to_numpy, contiguous_split_indices
from gpyro_prototype.loaders import (
    build_aligned_matrix,
    merge_aligned_bundles,
    read_coordinates,
    read_recorded_points,
    read_temperatures_corrected,
)
from gpyro_prototype.metrics import aggregate_dtw_mare_over_nodes, mae_rmse
from gpyro_prototype.model import ThermalMLP, build_features


def cap_rollout_horizon(bundle: dict, start_t: int, horizon: int) -> int:
    """Do not roll past the end of the experiment segment containing ``start_t``."""
    T = bundle["T"]
    ss = bundle.get("segment_starts")
    if ss is None or len(ss) < 2:
        return min(horizon, len(T) - start_t - 1)
    for i in range(len(ss) - 1):
        if ss[i] <= start_t < ss[i + 1]:
            return min(horizon, int(ss[i + 1] - start_t - 1))
    return min(horizon, len(T) - start_t - 1)


def autoregressive_rollout(
    model: nn.Module,
    bundle: dict,
    start_t: int,
    horizon: int,
    n_nodes: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Teacher-free rollout: feed predicted T back in.
    Returns (pred_stack, true_stack) each shape (horizon, n_nodes).
    """
    T = bundle["T"]
    horizon = cap_rollout_horizon(bundle, start_t, horizon)
    if start_t + horizon >= len(T):
        horizon = len(T) - start_t - 1
    if horizon < 1:
        raise ValueError("horizon too short for rollout")

    cur = torch.from_numpy(T[start_t].copy()).to(device).unsqueeze(0)
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for h in range(horizon):
            t = start_t + h
            feat = build_features(
                cur,
                torch.from_numpy(bundle["torch_x"][t : t + 1]).to(device),
                torch.from_numpy(bundle["torch_y"][t : t + 1]).to(device),
                torch.from_numpy(bundle["torch_z"][t : t + 1]).to(device),
                torch.from_numpy(bundle["torch_vx"][t : t + 1]).to(device),
                torch.from_numpy(bundle["torch_vy"][t : t + 1]).to(device),
                torch.from_numpy(bundle["torch_flag"][t : t + 1]).to(device),
                torch.from_numpy(bundle["boundary"][t : t + 1]).to(device),
            )
            nxt = model(feat)
            preds.append(nxt.squeeze(0).cpu().numpy())
            trues.append(T[t + 1].copy())
            cur = nxt
    return np.stack(preds, axis=0), np.stack(trues, axis=0)


def run_training(cfg: PrototypeConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recorded = read_recorded_points(cfg.recorded_points_path)
    bundles_single: list = []
    eids = list(cfg.experiment_ids)
    for eid in eids:
        temps = read_temperatures_corrected(cfg.temperatures_path(eid))
        coords = read_coordinates(cfg.coordinates_path(eid))
        bundles_single.append(build_aligned_matrix(temps, coords, recorded))
    bundle = bundle_to_numpy(merge_aligned_bundles(bundles_single, eids))
    n = len(bundle["T"])
    split = contiguous_split_indices(n, cfg.train_frac, cfg.val_frac)
    print(f"[data] experiments={eids}  merged_timesteps={n}")
    print(
        f"[split] train={split.train} val={split.val} test={split.test} "
        "(contiguous on merged timeline, cross-experiment pairs excluded)"
    )

    sample_feat, _ = ThermalSequenceDataset(bundle, split.train)[0]
    in_dim = int(sample_feat.shape[0])
    n_nodes = bundle["T"].shape[1]
    model = ThermalMLP(in_dim=in_dim, out_dim=n_nodes, hidden=cfg.hidden_dim).to(device)

    train_ds = ThermalSequenceDataset(bundle, split.train)
    val_ds = ThermalSequenceDataset(bundle, split.val)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(xb)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += float(loss_fn(pred, yb).item()) * len(xb)
        va_loss /= max(1, len(val_ds))
        print(f"epoch {ep:03d}  train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")

    # --- Evaluation on test segment: one-step + short rollout + paper-style DTW-MARE
    test_ds = ThermalSequenceDataset(bundle, split.test)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    model.eval()
    preds_one = []
    trues_one = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds_one.append(pred.cpu().numpy())
            trues_one.append(yb.numpy())
    y_pred_1 = np.concatenate(preds_one, axis=0)
    y_true_1 = np.concatenate(trues_one, axis=0)
    mae_1, rmse_1 = mae_rmse(y_true_1, y_pred_1)

    # Primary metric (paper): DTW-based mean absolute relative error (per node, then mean)
    dtw_mare, per_node = aggregate_dtw_mare_over_nodes(
        y_true_1,
        y_pred_1,
        eps_c=cfg.relative_eps_c,
        subsample_stride=cfg.dtw_subsample_stride,
    )

    # Rollout from first test index (stay inside one experiment segment)
    test_start = split.test.start or 0
    H = min(cfg.rollout_horizon, split.test.stop - test_start - 1)
    H = cap_rollout_horizon(bundle, test_start, H)
    pred_r, true_r = autoregressive_rollout(model, bundle, test_start, H, n_nodes, device)
    dtw_mare_roll, _ = aggregate_dtw_mare_over_nodes(
        true_r,
        pred_r,
        eps_c=cfg.relative_eps_c,
        subsample_stride=cfg.dtw_subsample_stride,
    )
    mae_r, rmse_r = mae_rmse(true_r, pred_r)

    print("\n=== Test (one-step, contiguous test region) ===")
    print(f"DTW-MARE (primary, paper-style): {dtw_mare * 100:.4f} %  (fraction {dtw_mare:.6f})")
    print(f"MAE (aux): {mae_1:.4f} °C   RMSE (aux): {rmse_1:.4f} °C")

    print(f"\n=== Test rollout (H={H} steps, dev horizon) ===")
    print(f"DTW-MARE (rollout): {dtw_mare_roll * 100:.4f} %")
    print(f"MAE (aux): {mae_r:.4f} °C   RMSE (aux): {rmse_r:.4f} °C")
    print(f"per-node DTW-MARE stats: min={per_node.min()*100:.3f}% max={per_node.max()*100:.3f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="GPyro development prototype trainer")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument(
        "--experiments",
        type=str,
        nargs="*",
        default=None,
        metavar="T1",
        help="Experiment ids to merge (default: T1..T10)",
    )
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--rollout", type=int, default=48)
    p.add_argument("--dtw-stride", type=int, default=8)
    args = p.parse_args()

    exp_ids: tuple[str, ...]
    if args.experiments:
        exp_ids = tuple(args.experiments)
    else:
        exp_ids = DEFAULT_EXPERIMENT_IDS
    cfg = PrototypeConfig(
        data_root=args.data_root,
        experiment_ids=exp_ids,
        epochs=args.epochs,
        rollout_horizon=args.rollout,
        dtw_subsample_stride=args.dtw_stride,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
