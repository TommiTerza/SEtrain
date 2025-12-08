#!/usr/bin/env python3
"""
Plot metric vs. occupancy trends from sweep.csv.

Usage example:
    python plot_sweep_occupancy.py --csv logs/threshold_opt/sweep.csv --out-dir logs/threshold_opt/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except Exception:
    adjust_text = None  # type: ignore[assignment]
    _HAS_ADJUST_TEXT = False


DEFAULT_OCC_COLUMNS = ("x_avg", "h_avg", "global_avg")
COMPARE_COLORS = ["red", "orange", "green", "violet", "blue", "magenta", "cyan", "brown"]
L1_BLOCK_COLUMNS = {
    "enc": [
        "enctra0_x",
        "enctra0_h",
        "enctra1_x",
        "enctra1_h",
        "enctra2_x",
        "enctra2_h",
    ],
    "dec": [
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


def _valid_columns(df: pd.DataFrame, names: Iterable[str]) -> list[str]:
    existing = []
    for name in names:
        if name not in df.columns:
            print(f"[plot] warning: column '{name}' not in CSV, skipping")
            continue
        existing.append(name)
    return existing


def _available_l1_blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for block, cols in L1_BLOCK_COLUMNS.items():
        present = [col for col in cols if col in df.columns]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            print(f"[l1] warning: missing columns for block '{block}': {', '.join(missing)}")
        if present:
            available[block] = present
    return available


def _l1_baseline_highlight_mask(df: pd.DataFrame, selected_block: str, block_columns: dict[str, list[str]]) -> list[bool]:
    if selected_block not in block_columns:
        available = ", ".join(sorted(block_columns)) or "none"
        raise ValueError(f"Block '{selected_block}' not available in CSV (available: {available})")
    other_columns = [col for block, cols in block_columns.items() if block != selected_block for col in cols]
    if not other_columns:
        return [False] * len(df)

    # enc uses the very first row; other blocks use the second row (baseline + inc on enc x)
    block_for_column = {col: block for block, cols in block_columns.items() for col in cols}
    if len(df) < 2:
        print("[l1] warning: fewer than 2 rows; using first row as baseline for all blocks")
    highlight_mask = pd.Series(True, index=df.index)
    for col in other_columns:
        block = block_for_column.get(col, None)
        baseline_idx = 0 if block == "enc" else 1
        if baseline_idx >= len(df):
            baseline_idx = len(df) - 1
        baseline_row = df.iloc[baseline_idx]
        base_value = baseline_row[col]
        series = df[col]
        if pd.isna(base_value):
            col_mask = series.isna()
        else:
            try:
                col_mask = pd.Series(np.isclose(series.astype(float).values, float(base_value)), index=df.index)
            except Exception:
                col_mask = series == base_value
        highlight_mask &= col_mask
    return highlight_mask.astype(bool).tolist()


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "nan"
    return f"{value:.3g}"


def _compute_pareto_mask(x: Sequence[float], y: Sequence[float], metric_higher_is_better: bool) -> list[bool]:
    n = len(x)
    mask = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            better_or_equal_occ = x[j] <= x[i]
            if metric_higher_is_better:
                better_or_equal_metric = y[j] >= y[i]
                strictly_better = (x[j] < x[i]) or (y[j] > y[i])
            else:
                better_or_equal_metric = y[j] <= y[i]
                strictly_better = (x[j] < x[i]) or (y[j] < y[i])
            if better_or_equal_occ and better_or_equal_metric and strictly_better:
                mask[i] = False
                break
    return mask


def _prepare_metric_dataframe(
    df: pd.DataFrame,
    metric_col: str,
    acc_deg: bool,
) -> tuple[pd.DataFrame, str, str, bool]:
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in CSV")
    metric_label = metric_col
    metric_higher_is_better = not acc_deg
    working_df = df.copy()
    if acc_deg:
        baseline = working_df[metric_col].max()
        if pd.isna(baseline) or baseline <= 0.0:
            raise ValueError(
                f"Cannot compute accuracy degradation: invalid baseline {baseline!r} "
                f"from column '{metric_col}'"
            )
        working_df["Accuracy degradation [%]"] = (baseline - working_df[metric_col]) / baseline * 100.0
        metric_col = "Accuracy degradation [%]"
        metric_label = f"{metric_label}_deg_pct"
        metric_higher_is_better = False
    return working_df, metric_col, metric_label, metric_higher_is_better


def _label_values_for_df(df: pd.DataFrame) -> list[str]:
    if "global" in df.columns:
        return [_format_float(v) for v in df["global"].values]
    if "threshold" in df.columns:
        return [_format_float(v) for v in df["threshold"].values]
    return [str(idx) for idx in range(len(df))]


def _common_valid_columns(
    dfs: Sequence[pd.DataFrame],
    names: Iterable[str],
    dataset_labels: Sequence[str],
) -> list[str]:
    common: list[str] = []
    for name in names:
        missing = [label for df, label in zip(dfs, dataset_labels) if name not in df.columns]
        if missing:
            print(f"[compare] warning: column '{name}' missing in {missing}, skipping")
            continue
        common.append(name)
    return common


def _plot_compare(
    datasets: Sequence[tuple[str, pd.DataFrame]],
    occupancy_col: str,
    metric_col: str,
    destination: Path,
    show_labels: bool,
    metric_higher_is_better: bool,
    interactive: bool,
    use_nice_labels: bool,
) -> None:
    combined_points: list[dict[str, object]] = []
    per_dataset_points: list[list[dict[str, object]]] = []

    for idx, (label, df) in enumerate(datasets):
        subset = df[[occupancy_col, metric_col]].copy()
        subset = subset.dropna(subset=[occupancy_col, metric_col])
        if subset.empty:
            raise ValueError(f"No valid rows for {occupancy_col} in '{label}'")

        base_labels = pd.Series(_label_values_for_df(df), index=df.index)
        subset["_label"] = base_labels.loc[subset.index].astype(str).values

        x = subset[occupancy_col].values
        y = subset[metric_col].values
        pareto_mask = _compute_pareto_mask(x, y, metric_higher_is_better)

        points: list[dict[str, object]] = []
        for xi, yi, lab, keep in zip(x, y, subset["_label"].values, pareto_mask):
            if not keep:
                continue
            entry = {"dataset": label, "x": xi, "y": yi, "label": str(lab)}
            points.append(entry)
            combined_points.append(entry)
        if not points:
            print(f"[compare] warning: no Pareto points for '{label}' on {occupancy_col}")
        per_dataset_points.append(points)

    if not combined_points:
        raise ValueError(f"No Pareto points to plot for {occupancy_col}")

    overall_mask = _compute_pareto_mask(
        [float(p["x"]) for p in combined_points],
        [float(p["y"]) for p in combined_points],
        metric_higher_is_better,
    )
    for entry, keep in zip(combined_points, overall_mask):
        entry["overall"] = bool(keep)

    plt.figure(figsize=(24, 12))
    for idx, points in enumerate(per_dataset_points):
        if not points:
            continue
        color = COMPARE_COLORS[idx % len(COMPARE_COLORS)]
        xs = [float(p["x"]) for p in points]
        ys = [float(p["y"]) for p in points]
        plt.scatter(xs, ys, color=color, edgecolors="black", linewidths=0.6, s=42, label=datasets[idx][0])

    if show_labels:
        texts = []
        for entry in combined_points:
            weight = "bold" if entry.get("overall") else "normal"
            label_text = f"{entry['dataset']}:{entry['label']}"
            texts.append(
                plt.text(
                    float(entry["x"]),
                    float(entry["y"]),
                    label_text,
                    fontsize=7,
                    alpha=0.8,
                    ha="left",
                    fontweight=weight,
                    bbox={"facecolor": "white", "edgecolor": "white", "alpha": 0.8, "pad": 1.5},
                )
            )
        if use_nice_labels and _HAS_ADJUST_TEXT:
            try:
                adjust_text(texts, only_move={"points": "y", "text": "y"})  # type: ignore[arg-type]
            except Exception as exc:
                print(f"[compare] adjustText failed ({exc}); proceeding with existing labels")

    plt.xlabel(f"{occupancy_col} (occupancy)")
    plt.ylabel(metric_col)
    plt.title(f"Pareto comparison: {metric_col} vs {occupancy_col}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    if interactive:
        plt.show()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(destination)
        plt.close()


def _plot_scatter(
    df: pd.DataFrame,
    occupancy_col: str,
    metric_col: str,
    destination: Path,
    derivative_destination: Path,
    labels: Optional[Sequence[str]] = None,
    draw_trend: bool = True,
    draw_derivative: bool = True,
    highlight_pareto: bool = False,
    metric_higher_is_better: bool = True,
    use_nice_labels: bool = False,
    interactive: bool = False,
    label_mask: Optional[Sequence[bool]] = None,
    always_label_pareto: bool = False,
    label_less: bool = False,
    show_labels: bool = True,
    highlight_mask: Optional[Sequence[bool]] = None,
    highlight_style: Optional[dict[str, object]] = None,
    highlight_label: Optional[str] = None,
    extra_highlights: Optional[Sequence[dict[str, object]]] = None,
) -> None:
    subset = df[[occupancy_col, metric_col]].copy()
    if labels is not None:
        subset = subset.assign(_label=list(labels))
    if label_mask is not None:
        subset = subset.assign(_label_mask=list(label_mask))
    highlight_columns: list[tuple[str, Optional[dict[str, object]], Optional[str]]] = []
    if highlight_mask is not None:
        subset = subset.assign(_highlight=list(highlight_mask))
        highlight_columns.append(("_highlight", highlight_style, highlight_label))
    if extra_highlights is not None:
        for idx, spec in enumerate(extra_highlights):
            if "mask" not in spec:
                raise ValueError("extra_highlights entries must include a 'mask' key")
            col_name = f"_highlight_extra_{idx}"
            subset = subset.assign(**{col_name: list(spec["mask"])})
            style = spec.get("style")
            label = spec.get("label")
            highlight_columns.append((col_name, style, label))
    subset = subset.dropna(subset=[occupancy_col, metric_col])
    if subset.empty:
        raise ValueError(f"No valid rows for {occupancy_col}")
    subset = subset.sort_values(occupancy_col)

    x = subset[occupancy_col].values
    y = subset[metric_col].values
    label_values = subset["_label"].astype(str).values if "_label" in subset.columns else None
    label_mask_values = subset["_label_mask"].astype(bool).values if "_label_mask" in subset.columns else None
    highlight_sets: list[tuple[np.ndarray, dict[str, object]]] = []
    if highlight_columns:
        for col_name, style, label in highlight_columns:
            if col_name not in subset.columns:
                continue
            mask_values = subset[col_name].astype(bool).values
            style_dict: dict[str, object] = {
                "s": 30,
                "color": "orange",
                "edgecolors": "black",
                "linewidths": 0.6,
            }
            if style:
                style_dict.update(style)
            if label is not None:
                style_dict.setdefault("label", label)
            elif "label" not in style_dict:
                style_dict.setdefault("label", "highlighted")
            highlight_sets.append((mask_values.astype(bool), style_dict))
    aggregated_highlight: Optional[np.ndarray] = None
    if highlight_sets:
        aggregated_highlight = np.logical_or.reduce([mask for mask, _ in highlight_sets])

    pareto_mask: Optional[list[bool]] = None
    if (highlight_pareto or always_label_pareto or label_less) and len(x) > 0:
        pareto_mask = _compute_pareto_mask(x, y, metric_higher_is_better)

    plt.figure(figsize=(24, 12))
    if pareto_mask is not None and highlight_pareto:
        non_front_x = [xi for xi, keep in zip(x, pareto_mask) if not keep]
        non_front_y = [yi for yi, keep in zip(y, pareto_mask) if not keep]
        front_x = [xi for xi, keep in zip(x, pareto_mask) if keep]
        front_y = [yi for yi, keep in zip(y, pareto_mask) if keep]
        if non_front_x:
            plt.scatter(non_front_x, non_front_y, s=30, label="runs")
        if front_x:
            plt.scatter(
                front_x,
                front_y,
                s=45,
                color="red",
                edgecolors="black",
                linewidths=0.5,
                label="Pareto front",
            )
    else:
        plt.scatter(x, y, s=30, label="runs")
    if highlight_sets:
        for mask_values, style in highlight_sets:
            highlight_x = [xi for xi, keep in zip(x, mask_values) if keep]
            highlight_y = [yi for yi, keep in zip(y, mask_values) if keep]
            if highlight_x:
                plt.scatter(highlight_x, highlight_y, **style)
    if label_values is not None and show_labels:
        if use_nice_labels and _HAS_ADJUST_TEXT:
            texts = []
            for idx, (xi, yi, lab) in enumerate(zip(x, y, label_values)):
                is_pareto = bool(pareto_mask[idx]) if pareto_mask is not None else False
                should_label = bool(label_mask_values[idx]) if label_mask_values is not None else True
                is_highlight = bool(aggregated_highlight[idx]) if aggregated_highlight is not None else False
                if label_less:
                    should_label = should_label and (is_pareto or is_highlight)
                if always_label_pareto and is_pareto:
                    should_label = True
                if not should_label:
                    continue
                ha = "right" if is_pareto else "left"
                weight = "bold" if (is_pareto and is_highlight) else "normal"
                texts.append(
                    plt.text(
                        xi,
                        yi,
                        lab,
                        fontsize=6,
                        alpha=0.7,
                        ha=ha,
                        fontweight=weight,
                        bbox={"facecolor": "white", "edgecolor": "white", "alpha": 0.8, "pad": 1.5},
                    )
                )
            try:
                adjust_text(texts, only_move={"points": "y", "text": "y"})  # type: ignore[arg-type]
            except Exception as exc:  # fallback to simple annotations on failure
                print(f"[plot] adjustText failed ({exc}); falling back to simple labels")
                for idx, (xi, yi, lab) in enumerate(zip(x, y, label_values)):
                    is_pareto = bool(pareto_mask[idx]) if pareto_mask is not None else False
                    should_label = bool(label_mask_values[idx]) if label_mask_values is not None else True
                    is_highlight = bool(aggregated_highlight[idx]) if aggregated_highlight is not None else False
                    if label_less:
                        should_label = should_label and (is_pareto or is_highlight)
                    if always_label_pareto and is_pareto:
                        should_label = True
                    if not should_label:
                        continue
                    offset = (-4, 2) if is_pareto else (2, 2)
                    ha = "right" if is_pareto else "left"
                    weight = "bold" if (is_pareto and is_highlight) else "normal"
                    plt.annotate(
                        lab,
                        (xi, yi),
                        textcoords="offset points",
                        xytext=offset,
                        fontsize=6,
                        alpha=0.7,
                        ha=ha,
                        fontweight=weight,
                        bbox={"facecolor": "white", "edgecolor": "white", "alpha": 0.8, "pad": 1.5},
                    )
        else:
            for idx, (xi, yi, lab) in enumerate(zip(x, y, label_values)):
                is_pareto = bool(pareto_mask[idx]) if pareto_mask is not None else False
                should_label = bool(label_mask_values[idx]) if label_mask_values is not None else True
                is_highlight = bool(aggregated_highlight[idx]) if aggregated_highlight is not None else False
                if label_less:
                    should_label = should_label and (is_pareto or is_highlight)
                if always_label_pareto and is_pareto:
                    should_label = True
                if not should_label:
                    continue
                offset = (-4, 2) if is_pareto else (2, 2)
                ha = "right" if is_pareto else "left"
                weight = "bold" if (is_pareto and is_highlight) else "normal"
                plt.annotate(
                    lab,
                    (xi, yi),
                    textcoords="offset points",
                    xytext=offset,
                    fontsize=6,
                    alpha=0.7,
                    ha=ha,
                    fontweight=weight,
                    bbox={"facecolor": "white", "edgecolor": "white", "alpha": 0.8, "pad": 1.5},
                )
    if draw_trend:
        plt.plot(x, y, color="C1", linewidth=1.0, alpha=0.7, label="sorted trend")
    plt.xlabel(f"{occupancy_col} (occupancy)")
    plt.ylabel(metric_col)
    plt.title(f"{metric_col} vs {occupancy_col}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    if interactive:
        plt.show()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(destination)
        plt.close()

    if draw_derivative and len(x) > 1:
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        derivative = dy / dx
        midpoints = 0.5 * (x[1:] + x[:-1])
        plt.figure(figsize=(24, 12))
        plt.plot(midpoints, derivative, linestyle="-", linewidth=1.0, color="C1", alpha=0.7)
        plt.scatter(midpoints, derivative, color="C0", s=16)
        plt.xlabel(f"{occupancy_col} midpoint")
        plt.ylabel(f"d({metric_col})/d{occupancy_col}")
        plt.title(f"Derivative of {metric_col} vs {occupancy_col}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if interactive:
            plt.show()
        else:
            plt.savefig(derivative_destination)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot occupancy vs metric from sweep.csv")
    parser.add_argument("--csv", type=Path, default=Path("logs/threshold_opt/sweep.csv"), help="Sweep CSV path")
    parser.add_argument("--metric-col", default="metric", help="Metric column name (default: metric)")
    parser.add_argument(
        "--occupancy-cols",
        nargs="*",
        default=list(DEFAULT_OCC_COLUMNS),
        help="Occupancy column names to plot (default: x_avg h_avg global_avg)",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "split", "per_gru", "l1", "compare"],
        default="global",
        help="Optimizer mode used to generate the sweep CSV (per_gru not implemented yet; l1 shows a compact overview).",
    )
    parser.add_argument(
        "--l1-block",
        choices=list(L1_BLOCK_COLUMNS.keys()),
        help="In l1 mode, highlight runs where all other blocks stay at baseline thresholds (from the first CSV row).",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs="+",
        type=Path,
        help="Directories to compare when using mode=compare (each must contain sweep.csv).",
    )
    parser.add_argument(
        "--acc-deg",
        action="store_true",
        help="Plot metric degradation in percent instead of raw metric "
             "(0%% = best metric; higher = worse).",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Highlight Pareto-optimal points (min occupancy, best metric) in red.",
    )
    parser.add_argument(
        "--nice-labels",
        action="store_true",
        help="Try to reduce label overlap using the adjustText library (if installed).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show plots interactively instead of saving PNG files.",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Do not draw run labels next to points.",
    )
    parser.add_argument(
        "--less-tag",
        action="store_true",
        help="Draw fewer labels (Pareto points, plus highlighted equal-threshold points in split overview).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/threshold_opt/plots"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    show_labels = not args.no_tag
    label_less = args.less_tag and show_labels

    if mode == "compare":
        if not args.compare_dirs or len(args.compare_dirs) < 2:
            raise ValueError("Compare mode requires at least two directories via --compare-dirs")
        compare_paths = [p.expanduser().resolve() for p in args.compare_dirs]
        missing_csv = [str(path) for path in compare_paths if not (path / "sweep.csv").is_file()]
        if missing_csv:
            raise FileNotFoundError(f"Missing sweep.csv in: {', '.join(missing_csv)}")

        dataset_labels: list[str] = []
        label_counts: dict[str, int] = {}
        datasets: list[tuple[str, pd.DataFrame]] = []
        metric_col: Optional[str] = None
        metric_label: Optional[str] = None
        metric_higher_is_better: Optional[bool] = None

        for path in compare_paths:
            base_label = path.name or str(path)
            count = label_counts.get(base_label, 0)
            label_counts[base_label] = count + 1
            label = base_label if count == 0 else f"{base_label}_{count}"
            df_raw = pd.read_csv(path / "sweep.csv")
            prepared_df, metric_col_candidate, metric_label_candidate, metric_higher_candidate = _prepare_metric_dataframe(
                df_raw,
                args.metric_col,
                args.acc_deg,
            )
            if metric_col is None:
                metric_col = metric_col_candidate
                metric_label = metric_label_candidate
                metric_higher_is_better = metric_higher_candidate
            elif metric_col != metric_col_candidate:
                raise ValueError(
                    f"Metric column mismatch across datasets: '{metric_col}' vs '{metric_col_candidate}'"
                )
            datasets.append((label, prepared_df))
            dataset_labels.append(label)

        assert metric_col is not None and metric_label is not None and metric_higher_is_better is not None
        occ_columns = _common_valid_columns([df for _, df in datasets], args.occupancy_cols, dataset_labels)
        if not occ_columns:
            raise ValueError("No occupancy columns available to plot for all compare datasets")

        compare_dir = output_dir / "compare"
        for col in occ_columns:
            path = compare_dir / f"{col}_vs_{metric_label}_compare.png"
            try:
                _plot_compare(
                    datasets,
                    col,
                    metric_col,
                    path,
                    show_labels=show_labels,
                    metric_higher_is_better=metric_higher_is_better,
                    interactive=args.interactive,
                    use_nice_labels=args.nice_labels,
                )
            except ValueError as exc:
                print(f"[compare] skipping {col}: {exc}")
            else:
                print(f"[compare] wrote {path}")
        return

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    df_raw = pd.read_csv(csv_path)
    df, metric_col, metric_label, metric_higher_is_better = _prepare_metric_dataframe(df_raw, args.metric_col, args.acc_deg)

    occ_columns = _valid_columns(df, args.occupancy_cols)
    if mode == "l1":
        # Keep the l1 overview focused on the core occupancy summaries
        occ_columns = _valid_columns(df, DEFAULT_OCC_COLUMNS)
    if not occ_columns:
        raise ValueError("No occupancy columns available to plot")

    if mode == "global":
        if "global" not in df.columns:
            raise ValueError("Global mode requires a 'global' column in the CSV")
        labels = [_format_float(v) for v in df["global"].values]
        for col in occ_columns:
            path = output_dir / f"{col}_vs_{metric_label}.png"
            deriv_path = output_dir / f"{col}_vs_{metric_label}_derivative.png"
            try:
                _plot_scatter(
                    df,
                    col,
                    metric_col,
                    path,
                    deriv_path,
                    labels=labels,
                    draw_trend=True,
                    draw_derivative=True,
                    highlight_pareto=args.pareto,
                    metric_higher_is_better=metric_higher_is_better,
                    use_nice_labels=args.nice_labels,
                    interactive=args.interactive,
                    show_labels=show_labels,
                    label_less=label_less,
                )
            except ValueError as exc:
                print(f"[plot] skipping {col}: {exc}")
            else:
                print(f"[plot] wrote {path}")
    elif mode == "split":
        if "global_x" not in df.columns or "global_h" not in df.columns:
            raise ValueError("Split mode requires 'global_x' and 'global_h' columns in the CSV")
        labels = [
            f"({_format_float(x)}, {_format_float(h)})"
            for x, h in zip(df["global_x"].values, df["global_h"].values)
        ]
        equal_threshold_mask = (
            df["global_x"].notna()
            & df["global_h"].notna()
            & np.isclose(df["global_x"].values, df["global_h"].values)
        )
        # Global scatter plots for all runs: scatter only, no trend/derivatives
        overview_dir = output_dir / "split_overview"
        for col in occ_columns:
            path = overview_dir / f"{col}_vs_{metric_label}_split.png"
            deriv_path = overview_dir / f"{col}_vs_{metric_label}_split_derivative.png"
            try:
                _plot_scatter(
                    df,
                    col,
                    metric_col,
                    path,
                    deriv_path,
                    labels=labels,
                    draw_trend=False,
                    draw_derivative=False,
                    highlight_pareto=args.pareto,
                    metric_higher_is_better=metric_higher_is_better,
                    use_nice_labels=args.nice_labels,
                    interactive=args.interactive,
                    show_labels=show_labels,
                    label_less=label_less,
                )
            except ValueError as exc:
                print(f"[plot] skipping {col} (split overview): {exc}")
            else:
                print(f"[plot] wrote {path}")

        # Overview with x==h highlighted and only Pareto/orange labels
        equal_overview_dir = output_dir / "split_overview_equal_thresholds"
        for col in occ_columns:
            path = equal_overview_dir / f"{col}_vs_{metric_label}_split_equal.png"
            deriv_path = equal_overview_dir / f"{col}_vs_{metric_label}_split_equal_derivative.png"
            try:
                _plot_scatter(
                    df,
                    col,
                    metric_col,
                    path,
                    deriv_path,
                    labels=labels,
                    label_mask=equal_threshold_mask.tolist(),
                    always_label_pareto=True,
                    highlight_mask=equal_threshold_mask.tolist(),
                    highlight_style={"s": 30, "color": "orange", "edgecolors": "black", "linewidths": 0.6},
                    highlight_label="x == h",
                    draw_trend=False,
                    draw_derivative=False,
                    highlight_pareto=args.pareto,
                    metric_higher_is_better=metric_higher_is_better,
                    use_nice_labels=args.nice_labels,
                    interactive=args.interactive,
                    show_labels=show_labels,
                    label_less=label_less,
                )
            except ValueError as exc:
                print(f"[plot] skipping {col} (split overview equal thresholds): {exc}")
            else:
                print(f"[plot] wrote {path}")

        # Grouped plots per unique h threshold, with trend + derivatives
        groups_root = output_dir / "split_by_h"
        grouped = df.dropna(subset=["global_h"]).groupby("global_h")
        for h_value, subset in grouped:
            h_tag = _format_float(h_value).replace("-", "m").replace(".", "p")
            h_dir = groups_root / f"h{h_tag}"
            group_labels = [
                f"x={_format_float(x)}, h={_format_float(h_value)}"
                for x in subset["global_x"].values
            ]
            for col in occ_columns:
                path = h_dir / f"{col}_vs_{metric_label}_h{h_tag}.png"
                deriv_path = h_dir / f"{col}_vs_{metric_label}_h{h_tag}_derivative.png"
                try:
                    _plot_scatter(
                        subset,
                        col,
                        metric_col,
                        path,
                        deriv_path,
                        labels=group_labels,
                        draw_trend=True,
                        draw_derivative=True,
                        highlight_pareto=args.pareto,
                        metric_higher_is_better=metric_higher_is_better,
                        use_nice_labels=args.nice_labels,
                        interactive=args.interactive,
                        show_labels=show_labels,
                        label_less=label_less,
                    )
                except ValueError as exc:
                    print(f"[plot] skipping {col} (h={h_value} group): {exc}")
                else:
                    print(f"[plot] wrote {path}")
    elif mode == "l1":
        labels = [str(idx) for idx in range(len(df))]
        highlight_mask = [idx == 0 for idx in range(len(df))]
        extra_highlights = None
        if args.l1_block:
            available_blocks = _available_l1_blocks(df)
            baseline_mask = _l1_baseline_highlight_mask(df, args.l1_block, available_blocks)
            extra_highlights = [
                {
                    "mask": baseline_mask,
                    "style": {"s": 32, "color": "gold", "edgecolors": "black", "linewidths": 0.7},
                    "label": f"{args.l1_block} sweep (others baseline)",
                }
            ]
        overview_dir = output_dir / "l1_overview"
        for col in occ_columns:
            path = overview_dir / f"{col}_vs_{metric_label}_l1.png"
            deriv_path = overview_dir / f"{col}_vs_{metric_label}_l1_derivative.png"
            try:
                _plot_scatter(
                    df,
                    col,
                    metric_col,
                    path,
                    deriv_path,
                    labels=labels,
                    draw_trend=False,
                    draw_derivative=False,
                    highlight_pareto=args.pareto,
                    metric_higher_is_better=metric_higher_is_better,
                    use_nice_labels=args.nice_labels,
                    interactive=args.interactive,
                    show_labels=show_labels,
                    label_less=False,
                    highlight_mask=highlight_mask,
                    highlight_style={"s": 36, "color": "green", "edgecolors": "black", "linewidths": 0.7},
                    highlight_label="start (idx 0)",
                    extra_highlights=extra_highlights,
                )
            except ValueError as exc:
                print(f"[plot] skipping {col} (l1 overview): {exc}")
            else:
                print(f"[plot] wrote {path}")
    else:
        raise NotImplementedError("per_gru mode is not implemented yet in plot_sweep_occupancy")


if __name__ == "__main__":
    main()
