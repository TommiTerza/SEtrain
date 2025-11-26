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


def _valid_columns(df: pd.DataFrame, names: Iterable[str]) -> list[str]:
    existing = []
    for name in names:
        if name not in df.columns:
            print(f"[plot] warning: column '{name}' not in CSV, skipping")
            continue
        existing.append(name)
    return existing


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
) -> None:
    subset = df[[occupancy_col, metric_col]].copy()
    if labels is not None:
        subset = subset.assign(_label=list(labels))
    if label_mask is not None:
        subset = subset.assign(_label_mask=list(label_mask))
    if highlight_mask is not None:
        subset = subset.assign(_highlight=list(highlight_mask))
    subset = subset.dropna(subset=[occupancy_col, metric_col])
    if subset.empty:
        raise ValueError(f"No valid rows for {occupancy_col}")
    subset = subset.sort_values(occupancy_col)

    x = subset[occupancy_col].values
    y = subset[metric_col].values
    label_values = subset["_label"].astype(str).values if "_label" in subset.columns else None
    label_mask_values = subset["_label_mask"].astype(bool).values if "_label_mask" in subset.columns else None
    highlight_mask_values = subset["_highlight"].astype(bool).values if "_highlight" in subset.columns else None

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
            plt.scatter(non_front_x, non_front_y, s=18, label="runs")
        if front_x:
            plt.scatter(
                front_x,
                front_y,
                s=30,
                color="red",
                edgecolors="black",
                linewidths=0.5,
                label="Pareto front",
            )
    else:
        plt.scatter(x, y, s=18, label="runs")
    if highlight_mask_values is not None:
        highlight_x = [xi for xi, keep in zip(x, highlight_mask_values) if keep]
        highlight_y = [yi for yi, keep in zip(y, highlight_mask_values) if keep]
        if highlight_x:
            style = {
                "s": 30,
                "color": "orange",
                "edgecolors": "black",
                "linewidths": 0.6,
            }
            if highlight_style:
                style.update(highlight_style)
            if highlight_label is not None:
                style.setdefault("label", highlight_label)
            else:
                style.setdefault("label", "highlighted")
            plt.scatter(highlight_x, highlight_y, **style)
    if label_values is not None and show_labels:
        if use_nice_labels and _HAS_ADJUST_TEXT:
            texts = []
            for idx, (xi, yi, lab) in enumerate(zip(x, y, label_values)):
                is_pareto = bool(pareto_mask[idx]) if pareto_mask is not None else False
                should_label = bool(label_mask_values[idx]) if label_mask_values is not None else True
                is_highlight = bool(highlight_mask_values[idx]) if highlight_mask_values is not None else False
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
                    is_highlight = bool(highlight_mask_values[idx]) if highlight_mask_values is not None else False
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
                is_highlight = bool(highlight_mask_values[idx]) if highlight_mask_values is not None else False
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
        choices=["global", "split", "per_gru"],
        default="global",
        help="Optimizer mode used to generate the sweep CSV (per_gru not implemented yet).",
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
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    df = pd.read_csv(csv_path)
    if args.metric_col not in df.columns:
        raise ValueError(f"Metric column '{args.metric_col}' not found in {csv_path}")
    metric_col = args.metric_col
    metric_label = metric_col
    metric_higher_is_better = not args.acc_deg
    if args.acc_deg:
        baseline = df[metric_col].max()
        if pd.isna(baseline) or baseline <= 0.0:
            raise ValueError(
                f"Cannot compute accuracy degradation: invalid baseline {baseline!r} "
                f"from column '{metric_col}'"
            )
        df["Accuracy degradation [%]"] = (baseline - df[metric_col]) / baseline * 100.0
        metric_col = "Accuracy degradation [%]"
        metric_label = f"{args.metric_col}_deg_pct"

    occ_columns = _valid_columns(df, args.occupancy_cols)
    if not occ_columns:
        raise ValueError("No occupancy columns available to plot")
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    show_labels = not args.no_tag
    label_less = args.less_tag and show_labels

    mode = args.mode
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
    else:
        raise NotImplementedError("per_gru mode is not implemented yet in plot_sweep_occupancy")


if __name__ == "__main__":
    main()
