from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _point_key(name: str) -> int:
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else 0


def sanitize_temperature_field(T: np.ndarray, t_min: float = 0.0, t_max: float = 2500.0) -> np.ndarray:
    """Forward-fill NaNs along time, clip physically implausible °C spikes (prototype guard)."""
    df = pd.DataFrame(T)
    df = df.ffill(axis=0).bfill(axis=0)
    arr = np.array(df.to_numpy(dtype=np.float32), copy=True)
    return np.clip(arr, t_min, t_max)


def _parse_european_float(token: str | float | int) -> float:
    if isinstance(token, (float, int)) and not isinstance(token, bool):
        return float(token)
    t = str(token).strip().strip('"').strip()
    if not t or t.lower() == "nan":
        return float("nan")
    # European decimal: single comma as decimal separator; dots rarely appear
    t = t.replace(" ", "")
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    elif "," in t and "." in t:
        # e.g. 1.234,56 vs 1,234.56 — data here uses ; fields with 29,375 style
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def read_temperatures_corrected(path: str | Path) -> pd.DataFrame:
    """
    Load semicolon-separated corrected temperatures.
    Handles comma-as-decimal, odd headers (Unnamed / BOM), and stray quotes.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty file: {path}")

    # Detect delimiter from header line
    header_line = lines[0]
    delim = ";" if header_line.count(";") >= header_line.count(",") else ","

    buf = io.StringIO(raw)
    # Try pandas first with explicit sep; fall back to manual parse if needed
    try:
        df = pd.read_csv(
            buf,
            sep=delim,
            engine="python",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        df = _manual_parse_semicolon_table(lines, delim)

    df = _normalize_temperature_columns(df)
    df = _apply_european_numeric_columns(df)
    return df


def _manual_parse_semicolon_table(lines: list[str], delim: str) -> pd.DataFrame:
    rows = []
    reader = csv.reader(lines, delimiter=delim)
    for row in reader:
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    header = [c.strip().strip('"') for c in rows[0]]
    data = rows[1:]
    return pd.DataFrame(data, columns=header)


def _normalize_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)
    # First two columns: time and id/frame
    if len(cols) >= 2:
        df.rename(columns={cols[0]: "time_s", cols[1]: "id_tag"}, inplace=True)
    point_cols = [c for c in df.columns if str(c).startswith("Point")]
    for c in ["time_s", "id_tag", *point_cols]:
        if c not in df.columns:
            continue
        df[c] = df[c].astype(str).str.strip().str.strip('"')
    return df


def _apply_european_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ("time_s", "id_tag"):
            out[c] = out[c].map(_parse_european_float)
        elif str(c).startswith("Point") or c.startswith("Unnamed"):
            out[c] = out[c].map(_parse_european_float)
    # Drop duplicate unnamed columns if all NaN
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def read_raw_temperature(path: str | Path) -> pd.DataFrame:
    """Comma-separated raw temperatures (uses dot decimals in this dataset)."""
    path = Path(path)
    df = pd.read_csv(path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    df = df.loc[:, ~df.columns.duplicated()]
    if "id_tag" not in df.columns and len(df.columns) >= 2:
        df.rename(columns={df.columns[0]: "index", df.columns[1]: "id_tag"}, inplace=True)
    point_cols = [c for c in df.columns if str(c).startswith("Point")]
    for c in point_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_coordinates(path: str | Path) -> pd.DataFrame:
    """
    Torch trajectory: x_mm, y_mm, z_mm, time_s, flag (0/1).
    No header in files observed; we assign stable names.
    Some files append trailing empty columns (e.g. ',,'); always take the first 5 fields.
    """
    path = Path(path)
    df = pd.read_csv(path, header=None, engine="python", on_bad_lines="skip")
    df = df.iloc[:, :5].copy()
    df.columns = ["x_mm", "y_mm", "z_mm", "time_s", "flag"]
    for c in ["x_mm", "y_mm", "z_mm", "time_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["flag"] = pd.to_numeric(df["flag"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["time_s"])
    df = df.sort_values("time_s").reset_index(drop=True)
    return df


def read_recorded_points(path: str | Path) -> pd.DataFrame:
    """
    Mapping from physical coordinates to pixel indices.
    Handles quoted header like \"X.mm,Y.mm,...\" on one line.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    # Normalize newlines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        # Strip outer quotes if the whole line is quoted
        if ln.startswith('"') and ln.endswith('"'):
            ln = ln[1:-1]
        reader = csv.reader([ln])
        row = next(reader)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        df[c] = df[c].map(lambda s: _parse_european_float(str(s)) if s not in (None, "") else np.nan)
    return df


def torch_trajectory_on_time(
    coords: pd.DataFrame,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate torch position onto temperature timeline (seconds).
    Before first / after last coordinate time: clamp to endpoints.
    """
    t_c = coords["time_s"].to_numpy(dtype=np.float64)
    x = coords["x_mm"].to_numpy(dtype=np.float64)
    y = coords["y_mm"].to_numpy(dtype=np.float64)
    z = coords["z_mm"].to_numpy(dtype=np.float64)
    flag = coords["flag"].to_numpy(dtype=np.float64)

    tq = np.asarray(time_s, dtype=np.float64)
    xi = np.interp(tq, t_c, x)
    yi = np.interp(tq, t_c, y)
    zi = np.interp(tq, t_c, z)
    fi = np.interp(tq, t_c, flag)
    return xi, yi, zi, fi


def boundary_features_from_grid(
    recorded: pd.DataFrame,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> np.ndarray:
    """
    Distance to domain bbox from Recorded_points (paper: plate geometry / boundary context).
    Returns shape (N, 4): dist to min_x, max_x, min_y, max_y edges (signed could be added later).
    """
    xcol = "X.mm" if "X.mm" in recorded.columns else next((c for c in recorded.columns if "X" in c), None)
    ycol = "Y.mm" if "Y.mm" in recorded.columns else next((c for c in recorded.columns if "Y" in c and "Pixel" not in c), None)
    if recorded.empty or xcol is None or ycol is None:
        return np.zeros((len(x_mm), 4), dtype=np.float32)

    xs = recorded[xcol].to_numpy()
    ys = recorded[ycol].to_numpy()
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))

    px = np.asarray(x_mm, dtype=np.float64)
    py = np.asarray(y_mm, dtype=np.float64)
    d_left = px - min_x
    d_right = max_x - px
    d_bot = py - min_y
    d_top = max_y - py
    return np.stack([d_left, d_right, d_bot, d_top], axis=1).astype(np.float32)


def build_aligned_matrix(
    temps: pd.DataFrame,
    coords: pd.DataFrame,
    recorded: pd.DataFrame,
) -> dict[str, Any]:
    """Merge temperature rows with interpolated torch path + boundary scalars."""
    time_s = temps["time_s"].to_numpy()
    point_cols = sorted([c for c in temps.columns if str(c).startswith("Point")], key=_point_key)
    T = temps[point_cols].to_numpy(dtype=np.float32)
    T = sanitize_temperature_field(T)

    xi, yi, zi, fi = torch_trajectory_on_time(coords, time_s)
    vx = np.gradient(xi, time_s, edge_order=1)
    vy = np.gradient(yi, time_s, edge_order=1)
    bfeat = boundary_features_from_grid(recorded, xi, yi)

    return {
        "time_s": time_s.astype(np.float32),
        "T": T,
        "point_cols": point_cols,
        "torch_x": xi.astype(np.float32),
        "torch_y": yi.astype(np.float32),
        "torch_z": zi.astype(np.float32),
        "torch_flag": fi.astype(np.float32),
        "torch_vx": vx.astype(np.float32),
        "torch_vy": vy.astype(np.float32),
        "boundary": bfeat,
    }


def merge_aligned_bundles(bundles: list[dict[str, Any]], experiment_ids: list[str]) -> dict[str, Any]:
    """
    Concatenate several single-experiment bundles along time.
    Sets ``segment_starts`` so callers can drop (t -> t+1) pairs that cross experiments.
    """
    if len(bundles) != len(experiment_ids):
        raise ValueError("bundles and experiment_ids length mismatch")
    if not bundles:
        raise ValueError("merge_aligned_bundles: empty list")

    keys = [
        "time_s",
        "T",
        "torch_x",
        "torch_y",
        "torch_z",
        "torch_vx",
        "torch_vy",
        "torch_flag",
        "boundary",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = np.concatenate([b[k] for b in bundles], axis=0)

    lengths = [len(b["T"]) for b in bundles]
    out["segment_starts"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    out["experiment_ids"] = list(experiment_ids)
    out["point_cols"] = bundles[0]["point_cols"]
    return out
