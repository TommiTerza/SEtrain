#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

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
    threshold: float,
    output_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for base in sorted(bases):
        for component in components:
            comp_arg = None if component == "all" else component
            try:
                stats = compute_occupancy(str(base), threshold, component=comp_arg)
            except FileNotFoundError:
                continue
            label = _label_for(base, component)
            plot_prefix = output_dir / label
            plot_analysis(stats, str(plot_prefix))
            summary = _summarize_metrics(stats)
            summary.update({"base": base.name, "component": component})
            rows.append(summary)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Batch GRU occupancy analysis.")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory containing *_x1.pkl/_h1.pkl logs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Delta threshold used for occupancy calculations.",
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

    logs_dir = Path(args.logs_dir).expanduser().resolve()
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory '{logs_dir}' does not exist.")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else logs_dir

    components = _ensure_components(args.components)
    bases = _discover_bases(logs_dir)
    if not bases:
        print(f"No GRU log files found in {logs_dir}.")
        return

    rows = _run_analysis(bases, components, args.threshold, output_dir)
    if not rows:
        print("No matching GRU components were analysed.")
        return

    summary_path = output_dir / "gru_occupancy_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)
    print(f"Wrote summary for {len(rows)} component(s) to {summary_path}")


if __name__ == "__main__":
    main()
