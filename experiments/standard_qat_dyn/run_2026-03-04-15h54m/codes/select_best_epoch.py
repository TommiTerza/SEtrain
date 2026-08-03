#!/usr/bin/env python3
"""
Post-training best-epoch selector with configurable criteria.

Supported metric sources:
1) TensorBoard logs under <run_dir>/logs
2) RESULTS files (default glob: */plots/scoring_intrusive/RESULTS.txt)

Examples:
  python select_best_epoch.py --run-dir experiments/standard_qat_no_distill/run_2026-03-03-10h38m \
      --source tensorboard --metric pesq --objective max

  python select_best_epoch.py --run-dir experiments/standard_qat_no_distill/run_2026-03-03-10h38m \
      --source tensorboard --metric val_loss --objective min

  python select_best_epoch.py --run-dir experiments/standard_qat_no_distill/run_2026-03-03-10h38m \
      --source results --expr "0.7*PESQ + 0.3*ESTOI" --objective max
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


MODEL_EPOCH_RE = re.compile(r"(?:^|_)model_(\d+)(?:$|\D)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
DEFAULT_RESULTS_GLOB = "*/plots/scoring_intrusive/RESULTS.txt"


def canonical_metric_name(name: str) -> str:
    metric = str(name).strip().lower()
    metric = NON_ALNUM_RE.sub("_", metric).strip("_")
    return metric


def parse_epoch_from_text(text: str) -> int | None:
    match = MODEL_EPOCH_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def parse_results_txt(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = canonical_metric_name(key)
        try:
            number = float(value.strip())
        except ValueError:
            continue
        if math.isfinite(number):
            metrics[key] = number
    return metrics


def find_epoch_for_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        epoch = parse_epoch_from_text(part)
        if epoch is not None:
            return epoch
    return None


def discover_checkpoint_epochs(run_dir: Path) -> set[int]:
    ckpt_dir = run_dir / "checkpoints"
    epochs: set[int] = set()
    if not ckpt_dir.exists():
        return epochs
    for path in ckpt_dir.glob("model_*.tar"):
        epoch = parse_epoch_from_text(path.name)
        if epoch is not None:
            epochs.add(epoch)
    return epochs


def checkpoint_for_epoch(run_dir: Path, epoch: int) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    primary = ckpt_dir / f"model_{epoch:03d}.tar"
    if primary.exists():
        return primary
    # Fallback to an existing best_model_XXX if the regular checkpoint is absent.
    fallback = ckpt_dir / f"best_model_{epoch:03d}.tar"
    if fallback.exists():
        return fallback
    return None


def _load_event_scalars(event_file: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard parser is unavailable. Install `tensorboard` to read event files."
        ) from exc

    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    out: dict[str, list[tuple[int, float]]] = {}

    for tag in accumulator.Tags().get("scalars", []):
        entries: list[tuple[int, float]] = []
        for event in accumulator.Scalars(tag):
            value = float(event.value)
            if math.isfinite(value):
                entries.append((int(event.step), value))
        if entries:
            out[tag] = entries
    return out


def _tensorboard_aliases(group: str, tag: str, from_root_event_file: bool) -> set[str]:
    aliases: set[str] = set()

    aliases.add(canonical_metric_name(group))
    if from_root_event_file:
        aliases.add(canonical_metric_name(tag))

    if from_root_event_file and "/" in tag:
        parts = [canonical_metric_name(part) for part in tag.split("/") if part]
        aliases.update(parts)
    if "_" in group:
        aliases.add(canonical_metric_name(group.split("_", 1)[1]))
        aliases.add(canonical_metric_name(group.rsplit("_", 1)[-1]))
    for prefix in ("val_loss_", "train_loss_", "lr_"):
        if group.startswith(prefix):
            aliases.add(canonical_metric_name(group[len(prefix) :]))

    aliases.discard("")
    return aliases


def load_tensorboard_records(run_dir: Path) -> tuple[dict[int, dict[str, float]], set[str]]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return {}, set()

    event_files = sorted(logs_dir.rglob("events.out.tfevents.*"))
    records: dict[int, dict[str, float]] = {}
    all_metrics: set[str] = set()

    for event_file in event_files:
        group = event_file.parent.name
        from_root_event_file = event_file.parent == logs_dir
        scalar_map = _load_event_scalars(event_file)
        for tag, series in scalar_map.items():
            aliases = _tensorboard_aliases(
                group=group,
                tag=tag,
                from_root_event_file=from_root_event_file,
            )
            all_metrics.update(aliases)
            for step, value in series:
                row = records.setdefault(step, {})
                for alias in aliases:
                    row[alias] = value

    return records, all_metrics


def load_results_records(
    run_dir: Path,
    results_glob: str,
) -> tuple[dict[int, dict[str, float]], set[str]]:
    records: dict[int, dict[str, float]] = {}
    all_metrics: set[str] = set()

    for result_file in sorted(run_dir.glob(results_glob)):
        if not result_file.is_file():
            continue
        epoch = find_epoch_for_path(result_file)
        if epoch is None:
            continue
        parsed = parse_results_txt(result_file)
        if not parsed:
            continue
        row = records.setdefault(epoch, {})
        row.update(parsed)
        all_metrics.update(parsed.keys())

    return records, all_metrics


def merge_records(
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {epoch: dict(values) for epoch, values in left.items()}
    for epoch, values in right.items():
        row = merged.setdefault(epoch, {})
        row.update(values)
    return merged


ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "clip": lambda x, lo, hi: max(lo, min(hi, x)),
}

ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
)


def compile_expression(expression: str) -> tuple[Any, set[str]]:
    tree = ast.parse(expression, mode="eval")

    variables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODE_TYPES):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("Only these functions are allowed in --expr: " + ", ".join(sorted(ALLOWED_FUNCS)))
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_FUNCS:
                variables.add(node.id)

    code = compile(tree, "<expr>", "eval")
    return code, variables


def evaluate_expression(
    code: Any,
    epoch: int,
    row: dict[str, float],
) -> float:
    env: dict[str, Any] = dict(ALLOWED_FUNCS)
    env["epoch"] = float(epoch)
    env["EPOCH"] = float(epoch)

    for key, value in row.items():
        env[key] = float(value)
        env[key.upper()] = float(value)

    result = eval(code, {"__builtins__": {}}, env)
    value = float(result)
    if not math.isfinite(value):
        raise ValueError("Expression returned a non-finite value.")
    return value


def select_source(
    source_arg: str,
    tb_records: dict[int, dict[str, float]],
    result_records: dict[int, dict[str, float]],
) -> tuple[str, dict[int, dict[str, float]]]:
    source_arg = source_arg.lower()

    if source_arg == "tensorboard":
        return "tensorboard", tb_records
    if source_arg == "results":
        return "results", result_records
    if source_arg == "both":
        return "both", merge_records(tb_records, result_records)

    # auto
    if len(result_records) >= 2:
        return "results", result_records
    if tb_records:
        return "tensorboard", tb_records
    return "results", result_records


def sort_key(rank_score: float, epoch: int, tie_break: str) -> tuple[float, int]:
    if tie_break == "latest":
        return rank_score, epoch
    # earliest
    return rank_score, -epoch


def maybe_create_link(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best epoch from post-training metrics.")
    parser.add_argument("--run-dir", required=True, help="Path to run directory (contains logs/ and checkpoints/).")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "tensorboard", "results", "both"],
        help="Metric source. `auto` prefers RESULTS if >=2 epochs are available.",
    )
    parser.add_argument(
        "--results-glob",
        default=DEFAULT_RESULTS_GLOB,
        help=f"Glob under run-dir for RESULTS files when source includes results. Default: {DEFAULT_RESULTS_GLOB}",
    )
    parser.add_argument("--metric", default="pesq", help="Metric used when --expr is not provided.")
    parser.add_argument("--objective", default="max", choices=["max", "min"], help="Optimize direction.")
    parser.add_argument(
        "--expr",
        default=None,
        help=(
            "Custom score expression (overrides --metric). "
            "Use metric names (e.g., PESQ, ESTOI, val_loss) and epoch. "
            "Allowed funcs: abs,min,max,sqrt,log,log10,exp,clip."
        ),
    )
    parser.add_argument("--epoch-min", type=int, default=None, help="Lower epoch bound (inclusive).")
    parser.add_argument("--epoch-max", type=int, default=None, help="Upper epoch bound (inclusive).")
    parser.add_argument("--tie-break", default="earliest", choices=["earliest", "latest"])
    parser.add_argument("--top-k", type=int, default=10, help="How many ranked epochs to print.")
    parser.add_argument(
        "--show-metrics",
        default="pesq,val_loss,sdr,sisnr,estoi",
        help="Comma-separated metric names to print in ranking rows.",
    )
    parser.add_argument(
        "--link-name",
        default=None,
        help="If set, create a symlink in checkpoints/ with this file name to the selected checkpoint.",
    )
    parser.add_argument(
        "--copy-name",
        default=None,
        help="If set, copy selected checkpoint to checkpoints/<copy-name>.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON file for selection details.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    tb_records, tb_metrics = load_tensorboard_records(run_dir)
    result_records, result_metrics = load_results_records(run_dir, args.results_glob)

    source_used, records = select_source(
        source_arg=args.source,
        tb_records=tb_records,
        result_records=result_records,
    )
    all_metrics = set(tb_metrics) | set(result_metrics)

    if not records:
        raise RuntimeError(
            "No metric records found. Check --source and --results-glob, "
            f"or verify logs under {run_dir / 'logs'}."
        )

    ckpt_epochs = discover_checkpoint_epochs(run_dir)

    candidate_epochs = sorted(records.keys())
    if ckpt_epochs:
        candidate_epochs = [epoch for epoch in candidate_epochs if epoch in ckpt_epochs]

    if args.epoch_min is not None:
        candidate_epochs = [epoch for epoch in candidate_epochs if epoch >= args.epoch_min]
    if args.epoch_max is not None:
        candidate_epochs = [epoch for epoch in candidate_epochs if epoch <= args.epoch_max]

    if not candidate_epochs:
        raise RuntimeError("No candidate epochs remain after filtering.")

    expr_code = None
    expr_variables: set[str] = set()
    metric_key = canonical_metric_name(args.metric)
    if args.expr:
        expr_code, expr_variables = compile_expression(args.expr)

    ranking: list[dict[str, Any]] = []
    skipped_epochs: list[int] = []
    for epoch in candidate_epochs:
        row = records.get(epoch, {})
        try:
            if expr_code is not None:
                objective_value = evaluate_expression(expr_code, epoch=epoch, row=row)
            else:
                if metric_key not in row:
                    skipped_epochs.append(epoch)
                    continue
                objective_value = float(row[metric_key])
        except Exception:
            skipped_epochs.append(epoch)
            continue

        rank_score = objective_value if args.objective == "max" else -objective_value
        ranking.append(
            {
                "epoch": epoch,
                "objective_value": objective_value,
                "rank_score": rank_score,
                "metrics": row,
            }
        )

    if not ranking:
        if args.expr:
            missing = sorted(canonical_metric_name(v) for v in expr_variables if canonical_metric_name(v) not in {"epoch"})
            raise RuntimeError(
                "No epochs could be scored from --expr. "
                f"Likely missing metrics. Expression variables: {sorted(expr_variables)}. "
                f"Available metrics: {sorted(all_metrics)}"
            )
        raise RuntimeError(
            f"Metric '{metric_key}' not available for candidate epochs. "
            f"Available metrics: {sorted(all_metrics)}"
        )

    ranking.sort(
        key=lambda item: sort_key(
            rank_score=float(item["rank_score"]),
            epoch=int(item["epoch"]),
            tie_break=args.tie_break,
        ),
        reverse=True,
    )

    best = ranking[0]
    best_epoch = int(best["epoch"])
    best_ckpt = checkpoint_for_epoch(run_dir, best_epoch)

    show_metrics = [canonical_metric_name(m) for m in args.show_metrics.split(",") if m.strip()]

    print(f"Run directory      : {run_dir}")
    print(f"Source used        : {source_used}")
    print(f"Candidate epochs   : {len(candidate_epochs)}")
    print(f"Skipped epochs     : {len(skipped_epochs)}")
    if args.expr:
        print(f"Criterion          : {args.objective} ({args.expr})")
    else:
        print(f"Criterion          : {args.objective}({metric_key})")
    print(f"Selected epoch     : {best_epoch}")
    print(f"Objective value    : {best['objective_value']:.6f}")
    if best_ckpt is not None:
        print(f"Selected checkpoint: {best_ckpt}")
    else:
        print("Selected checkpoint: <not found>")

    print("")
    print(f"Top {min(args.top_k, len(ranking))} epochs:")
    for item in ranking[: args.top_k]:
        epoch = int(item["epoch"])
        row = item["metrics"]
        parts = [f"epoch={epoch:03d}", f"obj={item['objective_value']:.6f}"]
        for metric in show_metrics:
            if metric in row:
                parts.append(f"{metric}={row[metric]:.6f}")
        print("  " + " | ".join(parts))

    if args.link_name:
        if best_ckpt is None:
            raise RuntimeError("Cannot create link because selected checkpoint file was not found.")
        link_name = args.link_name if args.link_name.endswith(".tar") else f"{args.link_name}.tar"
        link_path = run_dir / "checkpoints" / link_name
        maybe_create_link(link_path, best_ckpt)
        print(f"\nCreated symlink    : {link_path} -> {best_ckpt.name}")

    if args.copy_name:
        if best_ckpt is None:
            raise RuntimeError("Cannot copy because selected checkpoint file was not found.")
        copy_name = args.copy_name if args.copy_name.endswith(".tar") else f"{args.copy_name}.tar"
        copy_path = run_dir / "checkpoints" / copy_name
        shutil.copy2(best_ckpt, copy_path)
        print(f"Copied checkpoint  : {copy_path}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_dir": str(run_dir),
            "source": source_used,
            "objective": args.objective,
            "metric": None if args.expr else metric_key,
            "expr": args.expr,
            "selected": {
                "epoch": best_epoch,
                "objective_value": best["objective_value"],
                "checkpoint": str(best_ckpt) if best_ckpt is not None else None,
                "metrics": best["metrics"],
            },
            "top": [
                {
                    "epoch": int(item["epoch"]),
                    "objective_value": float(item["objective_value"]),
                    "metrics": item["metrics"],
                }
                for item in ranking[: args.top_k]
            ],
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote JSON report  : {output_path}")


if __name__ == "__main__":
    main()
