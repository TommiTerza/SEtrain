#!/usr/bin/env python
import argparse
import copy
import csv
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf
import yaml

from evaluate_gru_occupancy import compute_occupancy

RESULT_METRICS = {"SDR", "SISNR", "PESQ", "ESTOI"}
INTRUSIVE_METRIC = "intrusive"
PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(cmd: List[str], verbose: bool = False):
    if verbose:
        print(f"[opt] exec: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with code {proc.returncode}")


def read_results(result_path: Path) -> Dict[str, float]:
    results: Dict[str, float] = {}
    with result_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                results[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return results


@dataclass
class ThresholdParam:
    name: str
    getter: Callable[[dict], Optional[float]]
    setter: Callable[[dict, float], None]


class ConfigWriter:
    def __init__(self, path: Path):
        self.path = path
        self.base_data = yaml.safe_load(path.read_text())
        if "network_config" in self.base_data:
            self.section_key = "network_config"
        elif "network" in self.base_data:
            self.section_key = "network"
        else:
            raise ValueError(f"Config {path} missing 'network_config' or 'network' section")

    def render(self, thresholds: dict, log_file: Optional[Path] = None) -> str:
        data = copy.deepcopy(self.base_data)
        target = data[self.section_key]
        target["use_delta_gru"] = True
        target["delta_gru_threshold_x"] = thresholds["delta_gru_threshold_x"]
        target["delta_gru_threshold_h"] = thresholds["delta_gru_threshold_h"]
        target["delta_gru_thresholds"] = thresholds["delta_gru_thresholds"]
        if log_file is not None:
            target["log_gru_inputs"] = True
            target["log_file"] = str(log_file)
        return yaml.safe_dump(data, sort_keys=False)

    def write_to(self, dest: Path, thresholds: dict, log_file: Optional[Path] = None):
        dest.write_text(self.render(thresholds, log_file=log_file))


def build_initial_state(initial_value: float, mode: str) -> dict:
    def _block_list(count: int, value: Optional[float]) -> list:
        return [{"x": value, "h": value} for _ in range(count)]

    per_value = initial_value if mode == "per_gru" else None
    state = {
        "global_x": initial_value,
        "global_h": initial_value,
        "encoder_tra_blocks": _block_list(3, per_value if mode == "per_gru" else None),
        "decoder_tra_blocks": _block_list(3, per_value if mode == "per_gru" else None),
        "dpgrnn1": {name: {"x": per_value if mode == "per_gru" else None,
                             "h": per_value if mode == "per_gru" else None}
                     for name in ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2")},
        "dpgrnn2": {name: {"x": per_value if mode == "per_gru" else None,
                             "h": per_value if mode == "per_gru" else None}
                     for name in ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2")},
    }
    return state


def build_thresholds_dict(state: dict) -> dict:
    return {
        "delta_gru_threshold_x": state["global_x"],
        "delta_gru_threshold_h": state["global_h"],
        "delta_gru_thresholds": {
            "encoder_tra_blocks": copy.deepcopy(state["encoder_tra_blocks"]),
            "decoder_tra_blocks": copy.deepcopy(state["decoder_tra_blocks"]),
            "dpgrnn1": copy.deepcopy(state["dpgrnn1"]),
            "dpgrnn2": copy.deepcopy(state["dpgrnn2"]),
        },
    }


def make_parameters(mode: str, initial: float) -> List[ThresholdParam]:
    params: List[ThresholdParam] = []
    if mode == "global":
        params.append(ThresholdParam(
            "global",
            getter=lambda s: s["global_x"],
            setter=lambda s, v: (s.__setitem__("global_x", v), s.__setitem__("global_h", v)),
        ))
    elif mode == "split":
        params.append(ThresholdParam("global_x", getter=lambda s: s["global_x"], setter=lambda s, v: s.__setitem__("global_x", v)))
        params.append(ThresholdParam("global_h", getter=lambda s: s["global_h"], setter=lambda s, v: s.__setitem__("global_h", v)))
    elif mode == "per_gru":
        def add_block_params(section: str, count: int, prefix: str):
            for idx in range(count):
                params.append(ThresholdParam(
                    f"{prefix}{idx}_x",
                    getter=lambda s, section=section, idx=idx: s[section][idx]["x"],
                    setter=lambda s, v, section=section, idx=idx: s[section][idx].__setitem__("x", v),
                ))
                params.append(ThresholdParam(
                    f"{prefix}{idx}_h",
                    getter=lambda s, section=section, idx=idx: s[section][idx]["h"],
                    setter=lambda s, v, section=section, idx=idx: s[section][idx].__setitem__("h", v),
                ))

        add_block_params("encoder_tra_blocks", 3, "enc_tra_")
        add_block_params("decoder_tra_blocks", 3, "dec_tra_")
        for block in ("dpgrnn1", "dpgrnn2"):
            for comp in ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2"):
                params.append(ThresholdParam(
                    f"{block}_{comp}_x",
                    getter=lambda s, block=block, comp=comp: s[block][comp]["x"],
                    setter=lambda s, v, block=block, comp=comp: s[block][comp].__setitem__("x", v),
                ))
                params.append(ThresholdParam(
                    f"{block}_{comp}_h",
                    getter=lambda s, block=block, comp=comp: s[block][comp]["h"],
                    setter=lambda s, v, block=block, comp=comp: s[block][comp].__setitem__("h", v),
                ))
    else:
        raise ValueError(f"Unsupported mode {mode}")
    return params


def state_signature(state: dict) -> Tuple:
    def block_sig(entries: Iterable[dict]):
        return tuple((entry.get("x"), entry.get("h")) for entry in entries)
    return (
        state["global_x"],
        state["global_h"],
        block_sig(state["encoder_tra_blocks"]),
        block_sig(state["decoder_tra_blocks"]),
        tuple((state["dpgrnn1"][name]["x"], state["dpgrnn1"][name]["h"]) for name in ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2")),
        tuple((state["dpgrnn2"][name]["x"], state["dpgrnn2"][name]["h"]) for name in ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2")),
    )


class ThresholdOptimizer:
    def __init__(
        self,
        infer_config: Path,
        work_dir: Path,
        max_threshold: float,
        decay_factor: float,
        min_step: float,
        metric_name: str,
        mode: str,
        device: str,
        apply_best_to: Optional[Path] = None,
        verbose: bool = False,
        strategy: str = "coordinate",
        min_metric: float = 0.0,
        min_metric_drop: Optional[float] = None,
        occupancy_threshold: float = 1e-3,
        csv_output: Optional[Path] = None,
    ):
        self.infer_writer = ConfigWriter(infer_config)
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_threshold = max_threshold
        self.decay_factor = decay_factor
        self.min_step = min_step
        self.metric_name = metric_name
        self.mode = mode
        self.device = device
        self.apply_best_to = apply_best_to
        self.verbose = verbose
        self.strategy = strategy
        self.min_metric = min_metric
        self.min_metric_drop = min_metric_drop
        self.occupancy_threshold = occupancy_threshold
        self.csv_output = csv_output
        self.state = build_initial_state(0.0, mode)
        self.params = make_parameters(mode, 0.0)
        self.cache: Dict[Tuple, float] = {}
        self.best_metric: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.run_counter = 0

    def evaluate(self, log_base: Optional[Path] = None) -> float:
        sig = state_signature(self.state)
        if sig in self.cache:
            return self.cache[sig]
        cfg_dict = build_thresholds_dict(self.state)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=self.work_dir, delete=False) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(self.infer_writer.render(cfg_dict, log_file=log_base))
        try:
            if self.verbose:
                print(f"[opt] running infer.py with {temp_path.name}")
            run_command(["python", "infer.py", "-C", str(temp_path), "-D", self.device], verbose=self.verbose)
            if self.verbose:
                print(f"[opt] running evaluate.py for {temp_path.name}")
            run_command(["python", "evaluate.py", "--metric", INTRUSIVE_METRIC, "--config", str(temp_path)], verbose=self.verbose)
            cfg = OmegaConf.load(str(temp_path))
            results_path = Path(OmegaConf.to_container(cfg.network, resolve=True)["enh_folder"]) / "scoring_intrusive" / "RESULTS.txt"
            metrics = read_results(results_path)
            if self.metric_name not in metrics:
                raise RuntimeError(f"Metric {self.metric_name} missing in {results_path}")
            value = metrics[self.metric_name]
            if self.verbose:
                print(f"[opt] metric {self.metric_name} = {value:.4f}")
            self.cache[sig] = value
            return value
        finally:
            temp_path.unlink(missing_ok=True)

    def optimize_param(self, param: ThresholdParam, current_best: float) -> float:
        current_val = param.getter(self.state) or 0.0
        low = current_val
        high = self.max_threshold
        tol = 1e-6
        if self.verbose:
            print(f"[opt] optimizing {param.name} within [0, {self.max_threshold}] starting at {current_val}")
        best_val = current_val
        best_metric = current_best
        while (high - low) >= self.min_step:
            candidate = low + (high - low) * self.decay_factor
            candidate = max(candidate, low + self.min_step)
            if candidate > self.max_threshold:
                candidate = self.max_threshold
            param.setter(self.state, candidate)
            metric = self.evaluate()
            if self.verbose:
                print(f"[opt]   try {param.name}={candidate:.4f} -> {metric:.4f}")
            if metric > best_metric + tol:
                best_metric = metric
                best_val = candidate
                low = candidate
            else:
                high = candidate
        param.setter(self.state, best_val)
        return best_metric

    def collect_occupancy(self, log_base: Path) -> dict:
        components = ("x1", "x2", "h1", "h2", "x", "h")
        values: Dict[str, float] = {}
        for comp in components:
            try:
                stats = compute_occupancy(str(log_base), self.occupancy_threshold, component=comp)
                values[comp] = stats.get("mean_occupancy", 0.0)
            except Exception:
                continue
        summary = {f"occ_{comp}": values.get(comp) for comp in components}
        if values:
            occ_list = list(values.values())
            summary["occ_min"] = min(occ_list)
            summary["occ_max"] = max(occ_list)
            summary["occ_mean"] = sum(occ_list) / len(occ_list)
        else:
            summary["occ_min"] = summary["occ_max"] = summary["occ_mean"] = None
        return summary

    def run_coordinate(self):
        current = self.evaluate()
        for param in self.params:
            current = self.optimize_param(param, current)
        self.best_metric = current
        self.best_state = copy.deepcopy(self.state)
        if self.apply_best_to is not None:
            cfg_dict = build_thresholds_dict(self.state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def run_sweep(self):
        rows = []
        value = 0.0
        step = max(self.min_step, 1e-9)
        best_metric = -float("inf")
        best_state = None
        baseline_metric: Optional[float] = None
        while value <= self.max_threshold + 1e-9:
            log_base = self.work_dir / f"sweep_run_{self.run_counter}"
            self.run_counter += 1
            for param in self.params:
                param.setter(self.state, value)
            metric = self.evaluate(log_base=log_base)
            if baseline_metric is None:
                baseline_metric = metric
            occ = self.collect_occupancy(log_base)
            row = {"threshold": value, "metric": metric}
            row.update(occ)
            rows.append(row)
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.state)
            if self.verbose:
                print(f"[opt] sweep value {value} -> metric {metric:.4f}")
            stop = False
            if metric < self.min_metric:
                stop = True
            if baseline_metric is not None and self.min_metric_drop is not None:
                if metric < baseline_metric * (1.0 - self.min_metric_drop):
                    stop = True
            if stop:
                break
            value += step
        if rows and self.csv_output is not None:
            self.csv_output.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_output.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def run(self):
        if self.strategy == "coordinate":
            self.run_coordinate()
        elif self.strategy == "sweep":
            self.run_sweep()
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")


def main():
    parser = argparse.ArgumentParser(description="Optimize DeltaGRU thresholds via divide-and-conquer or sweep search")
    parser.add_argument("--infer-config", default="configs/cfg_infer.yaml", help="Inference config path")
    parser.add_argument("--train-config", default=None, help="Optional cfg_train.yaml to update with best thresholds")
    parser.add_argument("--mode", choices=["global", "split", "per_gru"], default="global")
    parser.add_argument("--strategy", choices=["coordinate", "sweep"], default="coordinate")
    parser.add_argument("--metric", choices=list(RESULT_METRICS), default="PESQ")
    parser.add_argument("--max-threshold", type=float, default=1.0, help="Upper bound for thresholds")
    parser.add_argument("--decay", type=float, default=0.5, help="Step decay factor (e.g. 0.5 for halving)")
    parser.add_argument("--min-step", type=float, default=0.05, help="Smallest step size before stopping")
    parser.add_argument("--work-dir", default="logs/threshold_opt", help="Working directory for temp configs")
    parser.add_argument("--device", default="0", help="GPU device id for infer.py")
    parser.add_argument("--min-metric", type=float, default=0.0, help="Stop sweep when metric falls below this value")
    parser.add_argument("--max-metric-drop", type=float, default=None, help="Relative drop (e.g. 0.15 for 15%) allowed vs baseline")
    parser.add_argument("--csv-output", default="logs/threshold_opt/sweep_results.csv", help="CSV file for sweep summaries")
    parser.add_argument("--occupancy-threshold", type=float, default=1e-3, help="Delta used for occupancy computation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    optimizer = ThresholdOptimizer(
        infer_config=Path(args.infer_config),
        work_dir=Path(args.work_dir).resolve(),
        max_threshold=args.max_threshold,
        decay_factor=args.decay,
        min_step=args.min_step,
        metric_name=args.metric,
        mode=args.mode,
        device=args.device,
        apply_best_to=Path(args.train_config) if args.train_config else None,
        verbose=args.verbose,
        strategy=args.strategy,
        min_metric=args.min_metric,
        min_metric_drop=args.max_metric_drop,
        occupancy_threshold=args.occupancy_threshold,
        csv_output=Path(args.csv_output).resolve() if args.csv_output else None,
    )
    optimizer.run()
    best = optimizer.best_metric if optimizer.best_metric is not None else float("nan")
    summary = {
        "metric": args.metric,
        "best_metric": best,
        "mode": args.mode,
        "max_threshold": args.max_threshold,
        "decay": args.decay,
        "best_state": optimizer.best_state,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
