#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

from delta_threshold_layout import RUN_CSV_NAME, RUN_DIR_PREFIX
from evaluate_gru_occupancy import compute_occupancy, plot_analysis, _summarize_metrics

SUFFIXES = ("_x1", "_x2", "_h1", "_h2", "_x", "_h")


def _discover_bases(directory: Path) -> set[Path]:
    bases: set[Path] = set()
    for path in directory.glob("*.pkl"):
        stem = path.stem
        matched = False
        for suffix in SUFFIXES:
            if stem.endswith(suffix):
                base_stem = stem[: -len(suffix)]
                bases.add(path.parent / base_stem)
                matched = True
                break
        if not matched:
            bases.add(path.with_suffix(""))
    return bases


def _ensure_components(raw: Sequence[str] | None) -> list[str]:
    if not raw:
        return ["all"]
    allowed = {"x1", "x2", "h1", "h2", "x", "h", "all"}
    components = []
    for comp in raw:
        comp = comp.lower()
        if comp not in allowed:
            raise ValueError(f"Unsupported component '{comp}'. Allowed values: {sorted(allowed)}")
        if comp == "all" and len(raw) > 1:
            continue
        components.append(comp)
    return components or ["all"]


def _label_for(base: Path, component: str) -> str:
    return f"{base.name}_{component}"


def _run_analysis(
    bases: Iterable[Path],
    components: list[str],
    threshold_lookup: Callable[[str, str], Optional[float]],
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for base in sorted(bases):
        base_name = base.name
        for component in components:
            thr = threshold_lookup(base_name, component)
            if thr is None:
                continue
            comp_arg = None if component == "all" else component
            try:
                stats = compute_occupancy(str(base), thr, component=comp_arg)
            except FileNotFoundError:
                continue
            label = _label_for(base, component)
            plot_prefix = output_dir / label
            plot_analysis(stats, str(plot_prefix))
            summary = _summarize_metrics(stats)
            summary.update({"base": base_name, "component": component, "threshold": thr})
            rows.append(summary)
    return rows


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _axis_for_component(component: str) -> Optional[str]:
    comp = component.lower()
    if comp.startswith("x"):
        return "x"
    if comp.startswith("h"):
        return "h"
    return None


def _load_run_row(run_dir: Path) -> dict:
    csv_path = run_dir / RUN_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Run CSV '{csv_path}' does not exist")
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    raise RuntimeError(f"Run CSV '{csv_path}' is empty")


def _build_threshold_lookup(row: dict, mode: str, fallback: Optional[float]) -> Callable[[str, str], Optional[float]]:
    def for_global(_: str, __: str) -> Optional[float]:
        value = _to_float(row.get("global"))
        return value if value is not None else fallback

    def for_split(_: str, component: str) -> Optional[float]:
        axis = _axis_for_component(component)
        if axis == "x":
            value = _to_float(row.get("global_x"))
        elif axis == "h":
            value = _to_float(row.get("global_h"))
        else:
            value = None
        if value is not None:
            return value
        # fall back to single global if available, then to CLI fallback
        value = _to_float(row.get("global"))
        return value if value is not None else fallback

    def for_per_gru(base: str, component: str) -> Optional[float]:
        axis = _axis_for_component(component)
        key = f"{base}_{axis}" if axis is not None else None
        value = _to_float(row.get(key)) if key is not None else None
        if value is not None:
            return value
        # fall back to split/global-style thresholds
        if axis == "x":
            value = _to_float(row.get("global_x"))
        elif axis == "h":
            value = _to_float(row.get("global_h"))
        if value is None:
            value = _to_float(row.get("global"))
        return value if value is not None else fallback

    if mode == "global":
        return for_global
    if mode == "split":
        return for_split
    if mode == "per_gru":
        return for_per_gru
    raise ValueError(f"Unsupported mode '{mode}' for threshold lookup")


def main():
    parser = argparse.ArgumentParser(description="Batch GRU occupancy analysis.")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Root directory for logs. When --run is set this should be the threshold optimizer work-dir (containing pkls/).",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Run index to analyse (expects pkls/run_<run>/thresholds.csv under logs-dir).",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "split", "per_gru"],
        default=None,
        help="Threshold layout for the selected run; required when --run is set.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Delta threshold used for occupancy calculations (or as fallback when using per-run CSV thresholds).",
    )
    parser.add_argument(
        "--components",
        nargs="*",
        help="Subset of components to analyse (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots and summary CSV (default: logs dir).",
    )
    args = parser.parse_args()

    root_dir = Path(args.logs_dir).expanduser().resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Logs directory '{root_dir}' does not exist.")

    if args.run is not None:
        if args.mode is None:
            parser.error("--mode is required when --run is specified")
        run_dir = root_dir / "pkls" / f"{RUN_DIR_PREFIX}{args.run}"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")
        row = _load_run_row(run_dir)
        threshold_lookup = _build_threshold_lookup(row, args.mode, args.threshold)
        logs_dir = run_dir
    else:
        threshold_lookup = lambda _base, _component: args.threshold
        logs_dir = root_dir

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else logs_dir

    components = _ensure_components(args.components)
    bases = _discover_bases(logs_dir)
    if not bases:
        print(f"No GRU log files found in {logs_dir}.")
        return

    rows = _run_analysis(bases, components, threshold_lookup, output_dir)
    if not rows:
        print("No matching GRU components were analysed.")
        return

    summary_path = output_dir / "gru_occupancy_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary for {len(rows)} component(s) to {summary_path}")


if __name__ == "__main__":
    main()
