#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from delta_threshold_layout import (
    COMPONENT_BASES,
    RUN_CSV_NAME,
    RUN_DIR_PREFIX,
    component_weight_map,
    parse_run_index,
    threshold_csv_columns,
)
try:
    from evaluate_gru_occupancy import compute_occupancy
except ModuleNotFoundError as exc:
    raise SystemExit(
        "summarize_run_occupancy requires the evaluation dependencies (numpy, matplotlib, pandas). "
        "Please install the project requirements before aggregating runs."
    ) from exc


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_run_row(run_dir: Path) -> Optional[dict]:
    csv_path = run_dir / RUN_CSV_NAME
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            return row
    return None


def _discover_components(run_dir: Path) -> Dict[str, set[str]]:
    base_map: Dict[str, set[str]] = {}
    for path in run_dir.glob("*.pkl"):
        stem = path.stem
        if stem.endswith("_x"):
            base = stem[:-2]
            comp = "x"
        elif stem.endswith("_h"):
            base = stem[:-2]
            comp = "h"
        else:
            continue
        base_map.setdefault(base, set()).add(comp)
    return base_map


def _weighted_mean(entries: List[Tuple[float, int, float]]) -> Optional[float]:
    if not entries:
        return None
    total_weight = 0.0
    total_value = 0.0
    for mean, samples, weight in entries:
        if samples <= 0 or weight <= 0:
            continue
        total_weight += samples * weight
        total_value += mean * samples * weight
    if total_weight > 0:
        return total_value / total_weight
    return sum(mean for mean, _, _ in entries) / len(entries)


def _build_threshold_lookup(row: dict, mode: str):
    if mode == "global":
        global_threshold = _to_float(row.get("global"))

        def lookup(_: str, __: str) -> Optional[float]:
            return global_threshold

        return lookup
    if mode == "split":
        threshold_x = _to_float(row.get("global_x"))
        threshold_h = _to_float(row.get("global_h"))

        def lookup(_: str, component: str) -> Optional[float]:
            return threshold_x if component == "x" else threshold_h

        return lookup
    per_component: Dict[Tuple[str, str], Optional[float]] = {}
    for base in COMPONENT_BASES:
        for comp in ("x", "h"):
            per_component[(base, comp)] = _to_float(row.get(f"{base}_{comp}"))

    def lookup(base: str, component: str) -> Optional[float]:
        return per_component.get((base, component))

    return lookup


def _process_run(
    run_dir: Path,
    row: dict,
    mode: str,
    default_threshold: Optional[float],
    weights: Dict[str, float],
    verbose: bool = False,
) -> dict:
    threshold_lookup = _build_threshold_lookup(row, mode)
    valid_bases = set(COMPONENT_BASES)
    entries = {"x": [], "h": []}
    for base, components in _discover_components(run_dir).items():
        if base not in valid_bases:
            continue
        for component in components:
            threshold = threshold_lookup(base, component)
            threshold_source = "csv"
            if threshold is None:
                threshold = default_threshold
                threshold_source = "fallback"
                if threshold is None:
                    if verbose:
                        print(f"[agg] {run_dir.name}/{base}_{component}: no threshold -> skipped")
                    continue
            try:
                stats = compute_occupancy(str(run_dir / base), threshold, component=component)
            except FileNotFoundError:
                if verbose:
                    print(f"[agg] missing logs for {base} component {component} in {run_dir.name}")
                continue
            except Exception as exc:
                if verbose:
                    print(f"[agg] failed to load {base} component {component}: {exc}")
                continue
            mean_occ = float(stats.get("mean_occupancy", 0.0))
            samples = int(stats.get("samples", 0) or 0)
            weight = weights.get(f"{base}_{component}", 1.0)
            entries[component].append((mean_occ, samples, weight))
            if verbose:
                print(
                    f"[agg] {run_dir.name}/{base}_{component}: "
                    f"thr={threshold:.4f} ({threshold_source}) mean={mean_occ:.4f} samples={samples} weight={weight}"
                )
    occ_x = _weighted_mean(entries["x"])
    occ_h = _weighted_mean(entries["h"])
    occ_global: Optional[float]
    if occ_x is not None and occ_h is not None:
        occ_global = 0.5 * (occ_x + occ_h)
    else:
        occ_global = occ_x if occ_x is not None else occ_h
    output = dict(row)
    output["x_avg"] = occ_x
    output["h_avg"] = occ_h
    output["global_avg"] = occ_global
    return output


def _iter_run_dirs(pkls_dir: Path) -> Iterable[Path]:
    entries: List[Tuple[int, Path]] = []
    for path in pkls_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(RUN_DIR_PREFIX):
            continue
        index, _ = parse_run_index(path)
        if index < 0:
            continue
        entries.append((index, path))
    entries.sort(key=lambda item: item[0])
    return [path for _, path in entries]


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-run GRU logs into a sweep CSV with occupancy stats")
    parser.add_argument("--pkls-dir", default="logs/threshold_opt/pkls", help="Directory containing run_* folders")
    parser.add_argument("--mode", choices=["global", "split", "per_gru"], required=True, help="Threshold layout to apply")
    parser.add_argument("--output", default="logs/threshold_opt/sweep.csv", help="Destination CSV for the sweep summary")
    parser.add_argument("--occupancy-threshold", type=float, default=1e-3,
                        help="Fallback delta for occupancy computation when a threshold is missing")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit the number of run folders to process")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pkls_dir = Path(args.pkls_dir).resolve()
    if not pkls_dir.exists():
        raise FileNotFoundError(f"pkls directory '{pkls_dir}' does not exist")

    weights = component_weight_map()
    rows: List[dict] = []
    processed = 0
    for run_dir in _iter_run_dirs(pkls_dir):
        if args.limit is not None and processed >= args.limit:
            break
        row = _read_run_row(run_dir)
        if row is None:
            if args.verbose:
                print(f"[agg] skipping {run_dir.name}: no {RUN_CSV_NAME}")
            continue
        if args.verbose:
            print(f"[agg] processing {run_dir.name}")
            rows.append(
                _process_run(
                    run_dir,
                    row,
                    args.mode,
                    args.occupancy_threshold,
                    weights,
                    verbose=args.verbose,
                )
            )
        processed += 1

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = threshold_csv_columns(include_metric=True) + ["x_avg", "h_avg", "global_avg"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.verbose:
        print(f"[agg] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
