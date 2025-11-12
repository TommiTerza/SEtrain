import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _load_component_arrays(log_file: str) -> dict[str, np.ndarray]:
    base, ext = os.path.splitext(log_file)
    ext = ext if ext else ".pkl"
    components = {}

    # Prefer the new multi-file format: <base>_{x1,h1,x2,h2,x,h}<ext>
    for key in ("x1", "h1", "x2", "h2", "x", "h"):
        candidate = f"{base}_{key}{ext}"
        if os.path.exists(candidate):
            with open(candidate, 'rb') as f:
                components[key] = pickle.load(f)

    if components:
        return components

    # Fallback to legacy single-file dictionary format
    if os.path.exists(log_file):
        with open(log_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            return data

    raise FileNotFoundError(
        f"Could not locate GRU input logs starting from base '{log_file}'."
    )


def _iter_sequences(array: np.ndarray):
    if array.ndim == 3:
        for seq in array:
            yield seq
    elif array.ndim == 2:
        yield array
    else:
        raise ValueError(f"Unsupported array shape {array.shape} for occupancy computation")


def _vector_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    if np.allclose(a, a.mean()) or np.allclose(b, b.mean()):
        # Degenerate variance; treat identical vectors as perfectly correlated
        return 1.0 if np.allclose(a, b) else 0.0
    return np.corrcoef(a, b)[0, 1]


def _aggregate_step_stats(sequences: Iterable[np.ndarray], threshold: float) -> dict:
    occupancy_total = 0.0
    occupancy_count = 0
    step_corr: dict[int, list[float]] = defaultdict(list)
    step_change_max: dict[int, list[float]] = defaultdict(list)
    step_change_mean: dict[int, list[float]] = defaultdict(list)
    step_change_min: dict[int, list[float]] = defaultdict(list)

    for seq in sequences:
        if seq.shape[0] < 2:
            continue
        diffs = seq[1:] - seq[:-1]
        abs_diff = np.abs(diffs)

        occupancy_total += np.mean(abs_diff > threshold)
        occupancy_count += 1

        for idx, (prev_vec, curr_vec, diff_vec) in enumerate(zip(seq[:-1], seq[1:], abs_diff)):
            corr = _vector_correlation(curr_vec, prev_vec)
            if not np.isnan(corr):
                step_corr[idx].append(float(corr))
            step_change_max[idx].append(float(diff_vec.max()))
            step_change_mean[idx].append(float(diff_vec.mean()))
            step_change_min[idx].append(float(diff_vec.min()))

    def _aggregate(metric: dict[int, list[float]]):
        steps = sorted(metric)
        values = np.array([np.mean(metric[idx]) for idx in steps]) if steps else np.array([])
        std = np.array([np.std(metric[idx]) for idx in steps]) if steps else np.array([])
        counts = np.array([len(metric[idx]) for idx in steps]) if steps else np.array([])
        return steps, values, std, counts

    steps_corr, corr_mean, corr_std, corr_counts = _aggregate(step_corr)
    steps_max, change_max_mean, change_max_std, change_max_counts = _aggregate(step_change_max)
    _, change_mean_mean, change_mean_std, _ = _aggregate(step_change_mean)
    _, change_min_mean, change_min_std, _ = _aggregate(step_change_min)

    mean_occupancy = occupancy_total / occupancy_count if occupancy_count else 0.0

    return {
        "mean_occupancy": mean_occupancy,
        "samples": occupancy_count,
        "correlation": {
            "steps": np.array(steps_corr, dtype=np.int32),
            "mean": corr_mean,
            "std": corr_std,
            "counts": corr_counts,
        },
        "change": {
            "steps": np.array(steps_corr, dtype=np.int32),
            "max": {
                "mean": change_max_mean,
                "std": change_max_std,
                "counts": change_max_counts,
            },
            "avg": {
                "mean": change_mean_mean,
                "std": change_mean_std,
                "counts": change_max_counts,
            },
            "min": {
                "mean": change_min_mean,
                "std": change_min_std,
                "counts": change_max_counts,
            },
        },
    }


def compute_occupancy(log_file, threshold, component=None):
    data = _load_component_arrays(log_file)

    if component is not None:
        if component not in data:
            raise KeyError(f"Requested component '{component}' not available in logs")
        items = {component: data[component]}
    else:
        items = data

    def _sequence_iter():
        for inputs in items.values():
            arrays = np.concatenate(inputs, axis=0) if isinstance(inputs, list) else np.asarray(inputs)
            for seq in _iter_sequences(arrays):
                yield np.asarray(seq)

    stats = _aggregate_step_stats(_sequence_iter(), threshold)
    return stats


def _plot_trend(x, y, ylabel, title, path, y_std=None):
    if x.size == 0 or y.size == 0:
        return None
    plt.figure(figsize=(8, 4))
    x_idx = x + 1  # shift to 1-based step for readability
    plt.plot(x_idx, y, label="mean")
    if y_std is not None and y_std.size == y.size:
        lower = y - y_std
        upper = y + y_std
        plt.fill_between(x_idx, lower, upper, color='b', alpha=0.2, label='±1 std')
    plt.xlabel('Step index (t-1 -> t)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if y_std is not None and y_std.size == y.size:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_analysis(stats: dict, prefix: str) -> dict[str, str]:
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
    paths: dict[str, str] = {}

    corr = stats.get("correlation", {})
    corr_path = _plot_trend(
        corr.get("steps", np.array([])),
        corr.get("mean", np.array([])),
        ylabel='Pearson correlation',
        title='Correlation between consecutive inputs',
        path=f"{prefix}_correlation.png",
        y_std=corr.get("std"),
    )
    if corr_path:
        paths['correlation'] = corr_path

    change = stats.get("change", {})
    steps = change.get("steps", np.array([]))
    if steps.size:
        plt.figure(figsize=(8, 4))
        x_idx = steps + 1
        for key, label in (("max", "Max change"), ("avg", "Average change"), ("min", "Min change")):
            series = change.get(key, {}).get("mean", np.array([]))
            if series.size:
                plt.plot(x_idx, series, label=label)
        plt.xlabel('Step index (t-1 -> t)')
        plt.ylabel('Absolute change')
        plt.title('Change magnitude between consecutive inputs')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        change_path = f"{prefix}_change.png"
        plt.savefig(change_path)
        plt.close()
        paths['change'] = change_path

    return paths


def _summarize_metrics(stats: dict) -> dict[str, float]:
    summary = {
        'mean_occupancy': stats.get('mean_occupancy', 0.0),
        'samples': stats.get('samples', 0),
    }

    corr = stats.get('correlation', {})
    if corr.get('mean') is not None and corr['mean'].size:
        summary['corr_mean'] = float(np.mean(corr['mean']))
        summary['corr_first'] = float(corr['mean'][0])

    change = stats.get('change', {})
    for key, label in (("max", "max_change"), ("avg", "avg_change"), ("min", "min_change")):
        series = change.get(key, {}).get('mean')
        if series is not None and series.size:
            summary[f'{label}_mean'] = float(np.mean(series))
            summary[f'{label}_first'] = float(series[0])
    return summary


def _print_table(rows: list[dict], title: str) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys() if k != 'label'})
    print(f"\n{title}")
    header = ['label'] + keys
    widths = [max(len(str(row.get(col, ''))) for row in rows + [{col: col}]) for col in header]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*header))
    for row in rows:
        values = [row.get('label', '')] + [row.get(col, '') for col in keys]
        print(fmt.format(*values))


def _save_table_as_image(rows: list[dict], title: str, path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df.set_index('label') if 'label' in df.columns else df
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.5), max(4, len(df) * 0.4)))
    ax.axis('off')
    tbl = ax.table(cellText=np.round(df.values, decimals=6), colLabels=df.columns, rowLabels=df.index,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _collect_directory_stats(directory: Path, threshold: float) -> dict[str, list[dict]]:
    suffixes = {'x1': '_x1', 'x2': '_x2', 'h1': '_h1', 'h2': '_h2', 'x': '_x', 'h': '_h'}
    tables: dict[str, list[dict]] = {key: [] for key in suffixes}

    for comp, suffix in suffixes.items():
        pattern = f"*{suffix}.pkl"
        for path in sorted(directory.glob(pattern)):
            stem = path.stem
            if not stem.endswith(suffix[1:]):
                continue
            base_stem = stem[:-len(suffix)]
            base_path = path.parent / base_stem
            try:
                stats = compute_occupancy(str(base_path), threshold, component=comp)
            except FileNotFoundError:
                continue
            summary = _summarize_metrics(stats)
            summary['label'] = f"{base_stem}_{comp}"
            tables[comp].append(summary)
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute GRU input occupancy mean from log file.')
    parser.add_argument('log_file', type=str, help='Path to GRU input log base (without component suffix)')
    parser.add_argument('--threshold', type=float, default=1e-3, help='Occupancy threshold')
    parser.add_argument('--component', choices=['x1', 'h1', 'x2', 'h2', 'x', 'h', 'all'], default='all',
                        help='Select a specific component to analyse (default: all)')
    parser.add_argument('--plot-prefix', type=str, default=None,
                        help='Prefix for output plots (default: same as log_file)')
    parser.add_argument('--table-all', action='store_true',
                        help='Summarize all matching x and h logs in the directory of log_file')
    args = parser.parse_args()
    log_path = Path(args.log_file).resolve()
    if log_path.is_dir():
        if not args.table_all:
            raise ValueError("log_file must be a base path, not a directory, unless --table-all is set")
        stats = None
    else:
        component = None if args.component == 'all' else args.component
        stats = compute_occupancy(str(log_path), args.threshold, component=component)
        print(f"Analysed {stats.get('samples', 0)} sequences")
        print(f"Mean occupancy rate: {stats.get('mean_occupancy', 0.0):.6f}")

        corr = stats.get('correlation', {})
        if corr.get('steps', np.array([])).size:
            first_corr = corr['mean'][0]
            print(f"Correlation mean (first step): {first_corr:.6f}")

        change = stats.get('change', {})
        if change.get('steps', np.array([])).size:
            for key, label in (("max", "Max"), ("avg", "Average"), ("min", "Min")):
                series = change.get(key, {}).get('mean')
                if series is not None and series.size:
                    print(f"{label} change (first step): {series[0]:.6f}")

        prefix = args.plot_prefix if args.plot_prefix is not None else str(log_path)
        paths = plot_analysis(stats, prefix)
        for label, path in paths.items():
            print(f"Saved {label} plot to {path}")

    if args.table_all:
        directory = log_path if log_path.is_dir() else log_path.parent
        tables = _collect_directory_stats(directory, args.threshold)
        x_components = tables['x1'] + tables['x2'] + tables['x']
        y_components = tables['h1'] + tables['h2'] + tables['h']
        x_rows = sorted(x_components, key=lambda row: row['label'])
        y_rows = sorted(y_components, key=lambda row: row['label'])
        _print_table(x_rows, 'X component summary (x1/x2/x)')
        _print_table(y_rows, 'Y component summary (h1/h2/h)')

        if x_rows:
            x_path = directory / 'summary_x_components.png'
            _save_table_as_image(x_rows, 'X component summary (x1/x2/x)', x_path)
            print(f"Saved X component table to {x_path}")
        if y_rows:
            y_path = directory / 'summary_y_components.png'
            _save_table_as_image(y_rows, 'Y component summary (h1/h2/h)', y_path)
            print(f"Saved Y component table to {y_path}")
