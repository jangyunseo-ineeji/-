from __future__ import annotations

import numpy as np


def dtw_accumulated_cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Classic DTW with L1 step cost |x_i - y_j|."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n, m = len(x), len(y)
    inf = np.inf
    D = np.full((n + 1, m + 1), inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = abs(x[i - 1] - y[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D


def dtw_best_path(D: np.ndarray) -> list[tuple[int, int]]:
    """
    Backtrack on accumulated DTW matrix D with shape (n+1, m+1), 1-based indexing
    for sequence indices (row i corresponds to x[i-1]).
    """
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        c_diag = D[i - 1, j - 1]
        c_up = D[i - 1, j]
        c_left = D[i, j - 1]
        m = min(c_diag, c_up, c_left)
        if m == c_diag:
            i, j = i - 1, j - 1
        elif m == c_up:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def dtw_mean_abs_relative_error(
    y_meas: np.ndarray,
    y_pred: np.ndarray,
    eps_c: float = 1.0,
    subsample_stride: int = 1,
) -> float:
    """
    Paper-style metric (Sideris et al., 2023):
    Align predicted and measured 1D series with DTW, then average
      |T_pred - T_meas| / max(|T_meas|, eps_c)
    over alignment pairs.

    `eps_c` plays the role of a floor in the denominator (°C) to avoid
    blow-ups on near-zero measured temperatures in real data.

    `subsample_stride` > 1 downsamples *both* series before DTW for dev speed.
    """
    ym = np.asarray(y_meas, dtype=np.float64).ravel()[:: max(1, int(subsample_stride))]
    yp = np.asarray(y_pred, dtype=np.float64).ravel()[:: max(1, int(subsample_stride))]
    if len(ym) < 2 or len(yp) < 2:
        return float(
            np.mean(np.abs(yp - ym) / np.maximum(np.abs(ym), eps_c))
        )

    D = dtw_accumulated_cost_matrix(yp, ym)
    path = dtw_best_path(D)
    rels = []
    for i, j in path:
        denom = max(abs(ym[j]), eps_c)
        rels.append(abs(yp[i] - ym[j]) / denom)
    return float(np.mean(rels)) if rels else 0.0


def aggregate_dtw_mare_over_nodes(
    meas: np.ndarray,
    pred: np.ndarray,
    eps_c: float = 1.0,
    subsample_stride: int = 1,
) -> tuple[float, np.ndarray]:
    """
    Per-node DTW-MARE then mean across nodes (paper reports distribution of node errors).

    meas, pred: shape (T, K) or (T, n_nodes)
    """
    meas = np.asarray(meas, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    assert meas.shape == pred.shape
    k = meas.shape[1]
    per_node = np.empty(k, dtype=np.float64)
    for ki in range(k):
        per_node[ki] = dtw_mean_abs_relative_error(
            meas[:, ki], pred[:, ki], eps_c=eps_c, subsample_stride=subsample_stride
        )
    return float(np.mean(per_node)), per_node


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse
