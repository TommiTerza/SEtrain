#!/usr/bin/env python3
"""
Trade-off analysis for sensitivity sweeps.

Given the CSV produced by `optimize_delta_thresholds.py --mode sensitivity --strategy sweep`,
this script now:
  * builds, for each block, a 2D grid over (th_x, th_h) of occupancy and accuracy (metric),
  * constructs smooth 3D surfaces from those grids via either
      - spline interpolation (RectBivariateSpline) or
      - Pareto-front filtering followed by interpolation,
  * estimates partial derivatives of occupancy and accuracy w.r.t th_x and th_h,
  * computes a trade-off ratio surface: |∇occupancy| / |∇accuracy|,
  * writes CSV grids (occupancy, accuracy, trade-off) and plots 3D surfaces per block.

CLI additions:
  * --method {spline, pareto} chooses the surface builder.
  * --block limits the analysis to a single block.
  * --occupancy-col optionally overrides the occupancy source column.
  * --metric-mode can use the absolute metric or its degradation vs. the block best (baseline - metric).
  * --output-mode can emit full trade-off maps or only the occupancy-vs-accuracy difference.

Notes:
  * Occupancy uses `global_avg`, `x_avg`, or `h_avg` when present; otherwise it falls back
    to the mean threshold for that block (proxy).
  * th_x/th_h values are derived from the mean of the block's *_x and *_h columns in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BLOCK_COLUMNS: Dict[str, List[str]] = {
    "encoder": [
        "enctra0_x",
        "enctra0_h",
        "enctra1_x",
        "enctra1_h",
        "enctra2_x",
        "enctra2_h",
    ],
    "decoder": [
        "dectra0_x",
        "dectra0_h",
        "dectra1_x",
        "dectra1_h",
        "dectra2_x",
        "dectra2_h",
    ],
    "dpgrnn1": [
        "dp1intra1_x",
        "dp1intra1_h",
        "dp1intra2_x",
        "dp1intra2_h",
        "dp1inter1_x",
        "dp1inter1_h",
        "dp1inter2_x",
        "dp1inter2_h",
    ],
    "dpgrnn2": [
        "dp2intra1_x",
        "dp2intra1_h",
        "dp2intra2_x",
        "dp2intra2_h",
        "dp2inter1_x",
        "dp2inter1_h",
        "dp2inter2_x",
        "dp2inter2_h",
    ],
}

OCCUPANCY_CANDIDATES = ("global_avg", "x_avg", "h_avg")


def _to_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _canonical_threshold(value: float, places: int = 6) -> float:
    """Round thresholds to a fixed precision to merge near-duplicates (avoids tiny deltas -> huge gradients)."""
    return round(value, places)


def _load_rows(csv_path: Path) -> Tuple[List[dict], List[str]]:
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    if "block" not in fieldnames:
        raise RuntimeError(f"Expected a 'block' column in {csv_path} (sensitivity sweep output)")
    if "metric" not in fieldnames:
        raise RuntimeError(f"Expected a 'metric' column in {csv_path}")
    return rows, fieldnames


def _compute_pareto_mask(x: Sequence[float], y: Sequence[float]) -> List[bool]:
    """Pareto front for minimizing x (occupancy) and maximizing y (metric)."""
    n = len(x)
    mask = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal_occ = x[j] <= x[i]
            better_or_equal_metric = y[j] >= y[i]
            strictly_better = (x[j] < x[i]) or (y[j] > y[i])
            if better_or_equal_occ and better_or_equal_metric and strictly_better:
                mask[i] = False
                break
    return mask


def _block_mean(row: dict, block: str, suffix: str) -> Optional[float]:
    cols = [c for c in BLOCK_COLUMNS.get(block, []) if c.endswith(suffix)]
    values = [_to_float(row.get(col)) for col in cols]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _mean_block_threshold(row: dict, block: str) -> Optional[float]:
    cols = BLOCK_COLUMNS.get(block, [])
    values = [_to_float(row.get(col)) for col in cols]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _compute_occupancy_value(row: dict, block: str, occupancy_col: Optional[str]) -> Optional[float]:
    if occupancy_col:
        return _to_float(row.get(occupancy_col))
    return _mean_block_threshold(row, block)


Point = Tuple[float, float, float, float]  # th_x, th_h, occupancy, accuracy


@dataclass
class BlockGrid:
    block: str
    x_values: List[float]
    h_values: List[float]
    occupancy: np.ndarray  # shape (len(x_values), len(h_values))
    accuracy: np.ndarray   # shape (len(x_values), len(h_values))
    points: List[Point]


def _collect_block_points(rows: Iterable[dict], block: str, occupancy_col: Optional[str]) -> List[Point]:
    points: List[Point] = []
    for row in rows:
        if row.get("block") != block:
            continue
        accuracy = _to_float(row.get("metric"))
        if accuracy is None:
            continue
        th_x = _block_mean(row, block, "_x")
        th_h = _block_mean(row, block, "_h")
        if th_x is None or th_h is None:
            continue
        th_x = _canonical_threshold(th_x)
        th_h = _canonical_threshold(th_h)
        occupancy = _compute_occupancy_value(row, block, occupancy_col)
        if occupancy is None:
            continue
        points.append((th_x, th_h, occupancy, accuracy))
    return points


def _build_block_grid(rows: Iterable[dict], block: str, occupancy_col: Optional[str], metric_mode: str) -> Optional[BlockGrid]:
    points = _collect_block_points(rows, block, occupancy_col)
    if not points:
        return None
    if metric_mode == "degradation":
        baseline = max(p[3] for p in points)
        points = [(tx, th, occ, baseline - acc if acc is not None else None) for tx, th, occ, acc in points]
    elif metric_mode != "absolute":
        raise ValueError(f"Unknown metric mode '{metric_mode}' (expected 'absolute' or 'degradation').")
    x_values = sorted({p[0] for p in points})
    h_values = sorted({p[1] for p in points})
    occ_grid = np.full((len(x_values), len(h_values)), np.nan, dtype=float)
    acc_grid = np.full_like(occ_grid, np.nan, dtype=float)
    buckets: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    x_index = {v: i for i, v in enumerate(x_values)}
    h_index = {v: j for j, v in enumerate(h_values)}
    for th_x, th_h, occ, acc in points:
        key = (x_index[th_x], h_index[th_h])
        buckets.setdefault(key, []).append((occ, acc))
    for (i, j), vals in buckets.items():
        occ_vals = [v[0] for v in vals]
        acc_vals = [v[1] for v in vals]
        occ_grid[i, j] = float(sum(occ_vals) / len(occ_vals))
        acc_grid[i, j] = float(sum(acc_vals) / len(acc_vals))
    return BlockGrid(
        block=block,
        x_values=x_values,
        h_values=h_values,
        occupancy=occ_grid,
        accuracy=acc_grid,
        points=points,
    )


def _fill_missing_with_nearest(grid: np.ndarray, x_values: Sequence[float], h_values: Sequence[float]) -> np.ndarray:
    filled = np.array(grid, copy=True, dtype=float)
    known_coords = np.argwhere(~np.isnan(filled))
    if known_coords.size == 0:
        raise RuntimeError("Grid is empty; no values available to fill missing entries.")
    known_x = np.array([x_values[i] for i, _ in known_coords])
    known_h = np.array([h_values[j] for _, j in known_coords])
    known_v = filled[known_coords[:, 0], known_coords[:, 1]]
    for i, j in np.argwhere(np.isnan(filled)):
        dx = known_x - x_values[i]
        dh = known_h - h_values[j]
        idx = int(np.argmin(dx * dx + dh * dh))
        filled[i, j] = known_v[idx]
    return filled


def _pareto_points(points: List[Point]) -> List[Point]:
    if not points:
        return []
    occ = [p[2] for p in points]
    acc = [p[3] for p in points]
    mask = _compute_pareto_mask(occ, acc)
    return [pt for pt, keep in zip(points, mask) if keep]


def _prepare_scattered_arrays(points: List[Point]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buckets: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
    for th_x, th_h, occ, acc in points:
        buckets.setdefault((th_x, th_h), []).append((occ, acc))
    xs: List[float] = []
    hs: List[float] = []
    occs: List[float] = []
    accs: List[float] = []
    for (th_x, th_h), vals in buckets.items():
        xs.append(th_x)
        hs.append(th_h)
        occs.append(sum(v[0] for v in vals) / len(vals))
        accs.append(sum(v[1] for v in vals) / len(vals))
    return np.array(xs), np.array(hs), np.array(occs), np.array(accs)


def _require_scipy():
    try:
        from scipy.interpolate import RectBivariateSpline, griddata
    except ImportError as exc:
        raise ImportError("scipy is required for --method spline/pareto (pip install scipy).") from exc
    return RectBivariateSpline, griddata


def _spline_surfaces(block_grid: BlockGrid) -> Dict[str, np.ndarray]:
    RectBivariateSpline, _ = _require_scipy()
    x_vals = np.array(block_grid.x_values, dtype=float)
    h_vals = np.array(block_grid.h_values, dtype=float)
    if len(x_vals) < 2 or len(h_vals) < 2:
        raise RuntimeError("Need at least 2 unique th_x and th_h values for spline interpolation.")
    occ_grid = _fill_missing_with_nearest(block_grid.occupancy, x_vals, h_vals)
    acc_grid = _fill_missing_with_nearest(block_grid.accuracy, x_vals, h_vals)
    kx = min(3, len(x_vals) - 1)
    ky = min(3, len(h_vals) - 1)
    occ_spline = RectBivariateSpline(x_vals, h_vals, occ_grid, kx=kx, ky=ky, s=0)
    acc_spline = RectBivariateSpline(x_vals, h_vals, acc_grid, kx=kx, ky=ky, s=0)

    occupancy = occ_spline(x_vals, h_vals)
    accuracy = acc_spline(x_vals, h_vals)
    docc_dx = np.gradient(occupancy, x_vals, axis=0)
    docc_dh = np.gradient(occupancy, h_vals, axis=1)
    dacc_dx = np.gradient(accuracy, x_vals, axis=0)
    dacc_dh = np.gradient(accuracy, h_vals, axis=1)
    return {
        "occupancy": occupancy,
        "accuracy": accuracy,
        "docc_dx": docc_dx,
        "docc_dh": docc_dh,
        "dacc_dx": dacc_dx,
        "dacc_dh": dacc_dh,
    }


def _interpolate_from_scattered(
    x_values: Sequence[float],
    h_values: Sequence[float],
    xs: np.ndarray,
    hs: np.ndarray,
    vals: np.ndarray,
    griddata,
) -> np.ndarray:
    X, H = np.meshgrid(h_values, x_values)
    points = np.column_stack((xs, hs))
    grid_linear = griddata(points, vals, (X, H), method="linear")
    grid_nearest = griddata(points, vals, (X, H), method="nearest")
    if grid_linear is None:
        return grid_nearest
    return np.where(np.isnan(grid_linear), grid_nearest, grid_linear)


def _pareto_surfaces(block_grid: BlockGrid) -> Dict[str, np.ndarray]:
    _, griddata = _require_scipy()
    pareto_pts = _pareto_points(block_grid.points)
    base_points = pareto_pts if len(pareto_pts) >= 3 else block_grid.points
    xs, hs, occs, accs = _prepare_scattered_arrays(base_points)
    if xs.size == 0:
        raise RuntimeError("No data points available for Pareto interpolation.")
    occupancy = _interpolate_from_scattered(block_grid.x_values, block_grid.h_values, xs, hs, occs, griddata)
    accuracy = _interpolate_from_scattered(block_grid.x_values, block_grid.h_values, xs, hs, accs, griddata)
    x_vals = np.array(block_grid.x_values, dtype=float)
    h_vals = np.array(block_grid.h_values, dtype=float)
    docc_dx = np.gradient(occupancy, x_vals, axis=0)
    docc_dh = np.gradient(occupancy, h_vals, axis=1)
    dacc_dx = np.gradient(accuracy, x_vals, axis=0)
    dacc_dh = np.gradient(accuracy, h_vals, axis=1)
    return {
        "occupancy": occupancy,
        "accuracy": accuracy,
        "docc_dx": docc_dx,
        "docc_dh": docc_dh,
        "dacc_dx": dacc_dx,
        "dacc_dh": dacc_dh,
    }


def _normalize_surface(values: np.ndarray, derivs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values, derivs
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    rng = vmax - vmin
    if rng <= 0:
        return values, derivs
    values_norm = (values - vmin) / rng
    derivs_norm = tuple(d / rng for d in derivs)
    return values_norm, derivs_norm


def _normalize_accuracy_to_baseline(values: np.ndarray, derivs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize accuracy so that the (0,0) threshold point becomes 1.0 and others are relative to it.
    Keeps relative slopes by scaling derivatives by the same baseline factor.
    """
    base = values[0, 0] if values.size else np.nan
    if not np.isfinite(base) or base == 0:
        return _normalize_surface(values, derivs)
    factor = float(base)
    values_norm = values / factor
    derivs_norm = tuple(d / factor for d in derivs)
    return values_norm, derivs_norm


def _normalize_field(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a single array independently."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    rng = vmax - vmin
    if rng <= 0:
        return values
    return (values - vmin) / rng


def _safe_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a-b where both are finite; otherwise NaN."""
    diff = np.full_like(a, np.nan)
    mask = np.isfinite(a) & np.isfinite(b)
    diff[mask] = a[mask] - b[mask]
    return diff


def _safe_sum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a+b where both are finite; otherwise NaN."""
    out = np.full_like(a, np.nan)
    mask = np.isfinite(a) & np.isfinite(b)
    out[mask] = a[mask] + b[mask]
    return out


def _safe_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a*b where both are finite; otherwise NaN."""
    out = np.full_like(a, np.nan)
    mask = np.isfinite(a) & np.isfinite(b)
    out[mask] = a[mask] * b[mask]
    return out


def _cap_tradeoff_slope(tradeoff: np.ndarray, max_slope: float = 1.0, passes: int = 2) -> np.ndarray:
    """
    Reduce local peaks so adjacent differences do not exceed max_slope.
    We iteratively pull down the larger of each adjacent pair (vertical and horizontal).
    """
    out = np.array(tradeoff, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return out
    for _ in range(max(1, passes)):
        # Vertical neighbors (along x axis)
        top = out[:-1, :]
        bottom = out[1:, :]
        mask = finite[:-1, :] & finite[1:, :]
        if mask.any():
            diff = bottom - top
            too_high = mask & (diff > max_slope)
            too_low = mask & (diff < -max_slope)
            bottom[too_high] = top[too_high] + max_slope
            top[too_low] = bottom[too_low] + max_slope
            out[:-1, :] = top
            out[1:, :] = bottom
        # Horizontal neighbors (along h axis)
        left = out[:, :-1]
        right = out[:, 1:]
        mask = finite[:, :-1] & finite[:, 1:]
        if mask.any():
            diff = right - left
            too_high = mask & (diff > max_slope)
            too_low = mask & (diff < -max_slope)
            right[too_high] = left[too_high] + max_slope
            left[too_low] = right[too_low] + max_slope
            out[:, :-1] = left
            out[:, 1:] = right
    return out


def _compute_tradeoff(block_grid: BlockGrid, method: str, normalize: bool, max_tradeoff: Optional[float], cap_unit_slope: bool) -> Dict[str, np.ndarray]:
    if method == "spline":
        surfaces = _spline_surfaces(block_grid)
    elif method == "pareto":
        surfaces = _pareto_surfaces(block_grid)
    else:
        raise ValueError(f"Unknown method '{method}' (expected 'spline' or 'pareto').")
    if normalize:
        surfaces["accuracy"], (surfaces["dacc_dx"], surfaces["dacc_dh"]) = _normalize_accuracy_to_baseline(
            surfaces["accuracy"], (surfaces["dacc_dx"], surfaces["dacc_dh"])
        )
    grad_occ = np.hypot(surfaces["docc_dx"], surfaces["docc_dh"])
    grad_acc = np.hypot(surfaces["dacc_dx"], surfaces["dacc_dh"])
    with np.errstate(divide="ignore", invalid="ignore"):
        tradeoff = grad_occ / grad_acc
        tradeoff_x = np.abs(surfaces["docc_dx"]) / np.abs(surfaces["dacc_dx"])
        tradeoff_h = np.abs(surfaces["docc_dh"]) / np.abs(surfaces["dacc_dh"])
    if cap_unit_slope:
        tradeoff = _cap_tradeoff_slope(tradeoff)
        tradeoff_x = _cap_tradeoff_slope(tradeoff_x)
        tradeoff_h = _cap_tradeoff_slope(tradeoff_h)
    if max_tradeoff is not None:
        tradeoff = np.where(np.isfinite(tradeoff), np.minimum(tradeoff, max_tradeoff), tradeoff)
        tradeoff_x = np.where(np.isfinite(tradeoff_x), np.minimum(tradeoff_x, max_tradeoff), tradeoff_x)
        tradeoff_h = np.where(np.isfinite(tradeoff_h), np.minimum(tradeoff_h, max_tradeoff), tradeoff_h)
    surfaces["tradeoff"] = tradeoff
    surfaces["tradeoff_x"] = tradeoff_x
    surfaces["tradeoff_h"] = tradeoff_h
    surfaces["grad_occ"] = grad_occ
    surfaces["grad_acc"] = grad_acc
    return surfaces


def _write_grid_csv(path: Path, x_values: Sequence[float], h_values: Sequence[float], grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["th_x/th_h"] + [f"{h:.6f}" for h in h_values]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, th_x in enumerate(x_values):
            row = [f"{th_x:.6f}"]
            for j, _ in enumerate(h_values):
                val = grid[i, j]
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    row.append("")
                else:
                    row.append(f"{float(val):.6g}")
            writer.writerow(row)


def _plot_surface(path: Path, x_values: Sequence[float], h_values: Sequence[float], grid: np.ndarray, title: str, zlabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting surfaces (pip install matplotlib).") from exc

    X, H = np.meshgrid(h_values, x_values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, H, grid, cmap=cm.viridis, edgecolor="none")
    ax.set_xlabel("th_h")
    ax.set_ylabel("th_x")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze sensitivity-mode sweep CSV and compute trade-off surfaces.")
    parser.add_argument("--csv", required=True, help="CSV produced by optimize_delta_thresholds.py in sensitivity mode.")
    parser.add_argument("--out-dir", default="logs/sensitivity_analysis", help="Directory to store outputs.")
    parser.add_argument("--occupancy-col", default=None, help="Optional explicit occupancy column to use (overrides auto-detection).")
    parser.add_argument("--method", choices=["spline", "pareto"], default="spline", help="Surface construction method.")
    parser.add_argument("--block", default=None, help="Optional block name to analyze (encoder/decoder/dpgrnn1/dpgrnn2).")
    parser.add_argument("--normalize", action="store_true", help="Min-max normalize occupancy and accuracy surfaces independently before gradients/trade-off.")
    parser.add_argument("--max-tradeoff", type=float, default=None, help="Optional cap applied to the trade-off surface to clip exploding ratios.")
    parser.add_argument("--cap-unit-slope", action="store_true", help="Scale trade-off so the maximum adjacent slope (abs delta) is at most 1.")
    parser.add_argument("--metric-mode", choices=["absolute", "degradation"], default="absolute", help="Use raw metric or degradation vs best (baseline - metric).")
    parser.add_argument(
        "--output-mode",
        choices=["tradeoff", "difference", "sum", "product"],
        default="tradeoff",
        help="Emit full trade-off maps, occupancy-accuracy difference, occupancy+accuracy sum, or occupancy*accuracy product.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    rows, fieldnames = _load_rows(csv_path)
    occupancy_col = args.occupancy_col
    if occupancy_col and occupancy_col not in fieldnames:
        raise RuntimeError(f"Requested occupancy column '{occupancy_col}' not found in CSV; available: {', '.join(fieldnames)}")
    if occupancy_col is None:
        occupancy_col = next((c for c in OCCUPANCY_CANDIDATES if c in fieldnames), None)
    if occupancy_col is None:
        print("[tradeoff] occupancy columns not found; falling back to mean threshold per block.")
    else:
        print(f"[tradeoff] using occupancy column '{occupancy_col}'")

    blocks = [args.block] if args.block else list(BLOCK_COLUMNS.keys())
    processed = 0
    for block in blocks:
        grid = _build_block_grid(rows, block, occupancy_col, args.metric_mode)
        if grid is None:
            print(f"[tradeoff] warning: no data for block '{block}'")
            continue
        try:
            surfaces = _compute_tradeoff(grid, args.method, args.normalize, args.max_tradeoff, args.cap_unit_slope)
        except Exception as exc:
            print(f"[tradeoff] error for block '{block}': {exc}")
            continue

        block_dir = out_dir / block
        block_dir.mkdir(parents=True, exist_ok=True)
        _write_grid_csv(block_dir / "occupancy_grid.csv", grid.x_values, grid.h_values, surfaces["occupancy"])
        _write_grid_csv(block_dir / "accuracy_grid.csv", grid.x_values, grid.h_values, surfaces["accuracy"])
        if args.output_mode == "tradeoff":
            _write_grid_csv(block_dir / "tradeoff_grid.csv", grid.x_values, grid.h_values, surfaces["tradeoff"])
            _write_grid_csv(block_dir / "tradeoff_x_grid.csv", grid.x_values, grid.h_values, surfaces["tradeoff_x"])
            _write_grid_csv(block_dir / "tradeoff_h_grid.csv", grid.x_values, grid.h_values, surfaces["tradeoff_h"])
        elif args.output_mode == "difference":
            diff = _safe_difference(surfaces["occupancy"], surfaces["accuracy"])
            _write_grid_csv(block_dir / "diff_occ_acc_grid.csv", grid.x_values, grid.h_values, diff)
        elif args.output_mode == "sum":
            summed = _safe_sum(surfaces["occupancy"], surfaces["accuracy"])
            sum_dx_raw = _safe_sum(surfaces["docc_dx"], surfaces["dacc_dx"])
            sum_dh_raw = _safe_sum(surfaces["docc_dh"], surfaces["dacc_dh"])
            summed_norm, (sum_dx_scaled, sum_dh_scaled) = _normalize_surface(summed, (sum_dx_raw, sum_dh_raw))
            sum_dx = _normalize_field(sum_dx_scaled)
            sum_dh = _normalize_field(sum_dh_scaled)
            grad_sum = np.hypot(sum_dx, sum_dh)
            _write_grid_csv(block_dir / "sum_occ_acc_grid.csv", grid.x_values, grid.h_values, summed_norm)
            _write_grid_csv(block_dir / "dsum_dx_grid.csv", grid.x_values, grid.h_values, sum_dx)
            _write_grid_csv(block_dir / "dsum_dh_grid.csv", grid.x_values, grid.h_values, sum_dh)
            _write_grid_csv(block_dir / "grad_sum_grid.csv", grid.x_values, grid.h_values, grad_sum)
        elif args.output_mode == "product":
            product = _safe_product(surfaces["occupancy"], surfaces["accuracy"])
            prod_dx_raw = surfaces["accuracy"] * surfaces["docc_dx"] + surfaces["occupancy"] * surfaces["dacc_dx"]
            prod_dh_raw = surfaces["accuracy"] * surfaces["docc_dh"] + surfaces["occupancy"] * surfaces["dacc_dh"]
            grad_prod = np.hypot(prod_dx_raw, prod_dh_raw)
            _write_grid_csv(block_dir / "product_occ_acc_grid.csv", grid.x_values, grid.h_values, product)
            _write_grid_csv(block_dir / "dproduct_dx_grid.csv", grid.x_values, grid.h_values, prod_dx_raw)
            _write_grid_csv(block_dir / "dproduct_dh_grid.csv", grid.x_values, grid.h_values, prod_dh_raw)
            _write_grid_csv(block_dir / "grad_product_grid.csv", grid.x_values, grid.h_values, grad_prod)
        _write_grid_csv(block_dir / "docc_dx_grid.csv", grid.x_values, grid.h_values, surfaces["docc_dx"])
        _write_grid_csv(block_dir / "docc_dh_grid.csv", grid.x_values, grid.h_values, surfaces["docc_dh"])
        _write_grid_csv(block_dir / "dacc_dx_grid.csv", grid.x_values, grid.h_values, surfaces["dacc_dx"])
        _write_grid_csv(block_dir / "dacc_dh_grid.csv", grid.x_values, grid.h_values, surfaces["dacc_dh"])
        _write_grid_csv(block_dir / "grad_occ_grid.csv", grid.x_values, grid.h_values, surfaces["grad_occ"])
        _write_grid_csv(block_dir / "grad_acc_grid.csv", grid.x_values, grid.h_values, surfaces["grad_acc"])

        try:
            _plot_surface(block_dir / "occupancy_surface.png", grid.x_values, grid.h_values, surfaces["occupancy"], f"{block} occupancy ({args.method})", "occupancy")
            _plot_surface(block_dir / "accuracy_surface.png", grid.x_values, grid.h_values, surfaces["accuracy"], f"{block} accuracy ({args.method})", "accuracy")
            if args.output_mode == "tradeoff":
                _plot_surface(block_dir / "tradeoff_surface.png", grid.x_values, grid.h_values, surfaces["tradeoff"], f"{block} trade-off ({args.method})", "trade-off")
                _plot_surface(block_dir / "tradeoff_x_surface.png", grid.x_values, grid.h_values, surfaces["tradeoff_x"], f"{block} trade-off x ({args.method})", "trade-off x")
                _plot_surface(block_dir / "tradeoff_h_surface.png", grid.x_values, grid.h_values, surfaces["tradeoff_h"], f"{block} trade-off h ({args.method})", "trade-off h")
            elif args.output_mode == "difference":
                diff = _safe_difference(surfaces["occupancy"], surfaces["accuracy"])
                _plot_surface(block_dir / "diff_occ_acc_surface.png", grid.x_values, grid.h_values, diff, f"{block} occ-acc diff ({args.method})", "occ - acc")
            elif args.output_mode == "sum":
                summed = _safe_sum(surfaces["occupancy"], surfaces["accuracy"])
                sum_dx_raw = _safe_sum(surfaces["docc_dx"], surfaces["dacc_dx"])
                sum_dh_raw = _safe_sum(surfaces["docc_dh"], surfaces["dacc_dh"])
                summed_norm, (sum_dx_scaled, sum_dh_scaled) = _normalize_surface(summed, (sum_dx_raw, sum_dh_raw))
                sum_dx = _normalize_field(sum_dx_scaled)
                sum_dh = _normalize_field(sum_dh_scaled)
                grad_sum = np.hypot(sum_dx, sum_dh)
                _plot_surface(block_dir / "sum_occ_acc_surface.png", grid.x_values, grid.h_values, summed_norm, f"{block} occ+acc sum ({args.method})", "occ + acc")
                _plot_surface(block_dir / "dsum_dx_surface.png", grid.x_values, grid.h_values, sum_dx, f"{block} d(sum)/dx ({args.method})", "d(sum)/dx")
                _plot_surface(block_dir / "dsum_dh_surface.png", grid.x_values, grid.h_values, sum_dh, f"{block} d(sum)/dh ({args.method})", "d(sum)/dh")
                _plot_surface(block_dir / "grad_sum_surface.png", grid.x_values, grid.h_values, grad_sum, f"{block} |∇(sum)| ({args.method})", "|∇(sum)|")
            elif args.output_mode == "product":
                product = _safe_product(surfaces["occupancy"], surfaces["accuracy"])
                prod_dx_raw = surfaces["accuracy"] * surfaces["docc_dx"] + surfaces["occupancy"] * surfaces["dacc_dx"]
                prod_dh_raw = surfaces["accuracy"] * surfaces["docc_dh"] + surfaces["occupancy"] * surfaces["dacc_dh"]
                grad_prod = np.hypot(prod_dx_raw, prod_dh_raw)
                _plot_surface(block_dir / "product_occ_acc_surface.png", grid.x_values, grid.h_values, product, f"{block} occ*acc product ({args.method})", "occ * acc")
                _plot_surface(block_dir / "dproduct_dx_surface.png", grid.x_values, grid.h_values, prod_dx_raw, f"{block} d(product)/dx ({args.method})", "d(product)/dx")
                _plot_surface(block_dir / "dproduct_dh_surface.png", grid.x_values, grid.h_values, prod_dh_raw, f"{block} d(product)/dh ({args.method})", "d(product)/dh")
                _plot_surface(block_dir / "grad_product_surface.png", grid.x_values, grid.h_values, grad_prod, f"{block} |∇(product)| ({args.method})", "|∇(product)|")
            _plot_surface(block_dir / "grad_occ_surface.png", grid.x_values, grid.h_values, surfaces["grad_occ"], f"{block} |∇occ| ({args.method})", "grad_occ")
            _plot_surface(block_dir / "grad_acc_surface.png", grid.x_values, grid.h_values, surfaces["grad_acc"], f"{block} |∇acc| ({args.method})", "grad_acc")
        except ImportError as exc:
            print(f"[tradeoff] plotting skipped for block '{block}': {exc}")

        processed += 1
        print(f"[tradeoff] block '{block}' processed -> {block_dir}")

    if processed == 0:
        print("[tradeoff] no blocks processed; check input CSV and filters.")


if __name__ == "__main__":
    main()
