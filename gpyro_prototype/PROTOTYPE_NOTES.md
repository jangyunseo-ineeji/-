# GPyro development prototype — scope and extensions

This folder is a **development prototype**, not a reproduction of Sideris et al. (2023), *Journal of Intelligent Manufacturing*.

## What is implemented

- Robust loaders for `temperatures/*_corrected.csv` (semicolon + European decimals), `raw/*.csv` (comma), `Coordinate_Time/Coordinates_*.csv`, and `Recorded_points.csv`.
- Alignment of torch trajectory onto the temperature timeline via `numpy.interp` (hold-endpoint outside the recorded path range).
- Contiguous **train / validation / test** splits along time (no shuffle); supervision pairs `(t → t+1)` never cross split boundaries.
- A small **MLP** that maps **[current temperature field ∥ torch pose/velocity/flag ∥ bbox distances]** to **next-step temperatures** (paper-style dynamical step; not the full GPyro architecture).
- **Primary metric**: per-node 1D DTW alignment, then mean absolute **relative** error vs `max(|T_meas|, ε)` (paper, Eq. around Sec. “Results”), with DTW input **subsampled** for dev speed (`dtw_subsample_stride`).
- **Secondary**: MAE / RMSE in °C.
- Short **autoregressive rollout** on the test segment (`rollout_horizon`) for qualitative error growth checks.

## Simplifications vs the paper (to extend later)

- No Gaussian-process time-varying parameters, no physics-informed split of dynamics, no Monte Carlo uncertainty bands.
- No full 27-experiment loop: default `experiment_id="T1"` only; extend by iterating `T1`…`T27` and aggregating metrics like the paper’s `Ê` distribution.
- DTW here is **per sensor node** (81 channels in this dataset vs 49 internal nodes in the paper); multivariate DTW over the full state vector is **not** used.
- Temperature sanitation: forward-fill/bfill along time + clip to **[0, 2500] °C** to suppress corrupted spikes (required after observing non-physical values in the corrected CSV).

## Data handling details (delimiters / headers / decimals)

- **Corrected temperatures**: primary separator **`;`**. Numeric fields often use **`,` as the decimal separator** (e.g. `29,375` → 29.375). Headers may be `Unnamed: 0` / odd quoting; parsers tolerate mixed pandas inference and fall back to manual CSV parsing.
- **Raw temperatures**: ordinary **comma** CSV with `.` decimals; some rows have irregular `id_tag` spacing — `on_bad_lines="skip"` guards parsing.
- **Coordinates**: no header in files; columns are fixed as **x, y, z [mm], time [s], flag**.
- **Recorded points**: lines may be wrapped in quotes; parsed with Python’s `csv` reader.

## How to run

From the repository root (`Reflow_Article`):

```bash
python -m gpyro_prototype.train --experiment T1 --epochs 15 --rollout 48 --dtw-stride 8
```

Adjust `--data-root` if `GPyro-TD` is not next to this package.
