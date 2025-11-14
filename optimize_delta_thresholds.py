#!/usr/bin/env python
import argparse
import copy
import csv
import json
import math
import subprocess
import tempfile
import time
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
        max_initial_failures: int = 0,
    ):
        self.infer_writer = ConfigWriter(infer_config)
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pickle_root = self.work_dir / "pkls"
        if self.pickle_root.exists():
            for item in self.pickle_root.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for sub in item.glob("*"):
                        sub.unlink(missing_ok=True)
                    item.rmdir()
        else:
            self.pickle_root.mkdir()
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
        self.threshold_columns = self._build_threshold_columns()
        self._component_prefixes: List[str] = []
        self._component_weights = self._build_component_weights()
        self.cache: Dict[Tuple, float] = {}
        self.best_metric: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.run_counter = 0
        self.max_initial_failures = max_initial_failures

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

    def _discover_log_bases(self, log_base: Path) -> Dict[Path, set[str]]:
        suffix_map = [
            ("_x1", "x1"),
            ("_x2", "x2"),
            ("_h1", "h1"),
            ("_h2", "h2"),
            ("_x", "x"),
            ("_h", "h"),
        ]
        log_dir = log_base.parent
        if not log_dir.exists():
            return {}
        prefix = log_base.stem
        bases: Dict[Path, set[str]] = {}
        for path in log_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            if not stem.startswith(prefix):
                continue
            remainder = stem[len(prefix):]
            if remainder and remainder[0] not in ("_", "."):
                continue
            for suffix, component in suffix_map:
                if not stem.endswith(suffix):
                    continue
                base_stem = stem[:-len(suffix)]
                ext = path.suffix
                base_name = f"{base_stem}{ext}" if ext else base_stem
                base_path = path.parent / base_name
                bases.setdefault(base_path, set()).add(component)
                break
        return bases

    def collect_occupancy(self, log_base: Path) -> dict:
        components = ("x", "h")
        per_component: Dict[str, List[Tuple[float, int, float]]] = {comp: [] for comp in components}
        base_map = self._discover_log_bases(log_base)
        if not base_map:
            base_map = {log_base: set()}
        for base, available_components in base_map.items():
            target_components = sorted(available_components) if available_components else ()
            if not target_components:
                continue
            for comp in target_components:
                try:
                    stats = compute_occupancy(str(base), self.occupancy_threshold, component=comp)
                except Exception as exc:
                    if self.verbose:
                        print(f"[opt] failed to read occupancy for {base} component {comp}: {exc}")
                    continue
                mean_occ = float(stats.get("mean_occupancy", 0.0))
                samples_val = stats.get("samples", 0)
                try:
                    samples = int(samples_val)
                except (TypeError, ValueError):
                    samples = 0
                key = self._component_key(base, comp)
                weight = self._component_weights.get(key, 1.0)
                bucket = "x" if comp.startswith("x") else "h"
                per_component[bucket].append((mean_occ, samples, weight))
                if self.verbose:
                    label = f"{base.stem} component {comp}"
                    if samples > 0:
                        print(f"[opt] occupancy for {label}: {mean_occ:.4f} ({samples} seqs)")
                    else:
                        print(f"[opt] occupancy for {label}: {mean_occ:.4f}")
        summary: Dict[str, Optional[float]] = {"occ_x": None, "occ_y": None, "occ_tot": None}

        def _weighted_mean(entries: List[Tuple[float, int, float]]) -> Optional[float]:
            if not entries:
                return None
            total_weight = 0.0
            accum = 0.0
            for mean, samples, weight in entries:
                if samples <= 0 or weight <= 0:
                    continue
                total_weight += samples * weight
                accum += mean * samples * weight
            if total_weight > 0:
                return accum / total_weight
            # fallback: equal-weight average
            return sum(mean for mean, _, _ in entries) / len(entries)

        occ_x = _weighted_mean(per_component["x"])
        occ_h = _weighted_mean(per_component["h"])
        if occ_x is not None:
            summary["occ_x"] = occ_x
        if occ_h is not None:
            summary["occ_y"] = occ_h
        if occ_x is not None and occ_h is not None:
            summary["occ_tot"] = 0.5 * (occ_x + occ_h)
        elif occ_x is not None:
            summary["occ_tot"] = occ_x
        elif occ_h is not None:
            summary["occ_tot"] = occ_h
        else:
            if self.verbose:
                print(f"[opt] warning: no GRU occupancy data found for {log_base}")
        return summary

    def _build_component_weights(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        prefixes: List[str] = []
        # Encoder TRA blocks
        for idx in range(3):
            prefix = f"enc_tra{idx}"
            prefixes.append(prefix)
            weights[f"{prefix}_x"] = 8.0
            weights[f"{prefix}_h"] = 16.0
        # Decoder TRA blocks
        for idx in range(3):
            prefix = f"dec_tra{idx}"
            prefixes.append(prefix)
            weights[f"{prefix}_x"] = 8.0
            weights[f"{prefix}_h"] = 16.0
        # Dual-path GRNN components
        for dp in ("dp1", "dp2"):
            for stage in ("intra", "inter"):
                prefix = f"{dp}_{stage}"
                prefixes.append(prefix)
                weights[f"{prefix}_x1"] = 8.0
                weights[f"{prefix}_x2"] = 8.0
                weights[f"{prefix}_h1"] = 8.0
                weights[f"{prefix}_h2"] = 8.0
        # Default fallbacks when prefixes do not match
        for comp in ("x", "h", "x1", "x2", "h1", "h2"):
            weights.setdefault(comp, 1.0)
        self._component_prefixes = prefixes
        return weights

    def _build_threshold_columns(self) -> List[str]:
        columns = ["threshold_global", "threshold_x", "threshold_h"]
        columns.extend(self._block_column_names("encoder_tra", len(self.state["encoder_tra_blocks"])))
        columns.extend(self._block_column_names("decoder_tra", len(self.state["decoder_tra_blocks"])))
        dp_components = ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2")
        columns.extend(self._dict_column_names("dpgrnn1", dp_components))
        columns.extend(self._dict_column_names("dpgrnn2", dp_components))
        return columns

    @staticmethod
    def _block_column_names(prefix: str, count: int) -> List[str]:
        cols = []
        for idx in range(count):
            cols.append(f"{prefix}_{idx}_x")
            cols.append(f"{prefix}_{idx}_h")
        return cols

    @staticmethod
    def _dict_column_names(prefix: str, components: Sequence[str]) -> List[str]:
        cols = []
        for name in components:
            cols.append(f"{prefix}_{name}_x")
            cols.append(f"{prefix}_{name}_h")
        return cols

    def _threshold_config_row(self) -> Dict[str, Optional[float]]:
        row: Dict[str, Optional[float]] = {col: None for col in self.threshold_columns}
        if self.mode == "global":
            row["threshold_global"] = self.state["global_x"]
            return row
        if self.mode == "split":
            row["threshold_x"] = self.state["global_x"]
            row["threshold_h"] = self.state["global_h"]
            return row
        if self.mode != "per_gru":
            return row
        self._fill_block_thresholds(row, "encoder_tra_blocks", "encoder_tra")
        self._fill_block_thresholds(row, "decoder_tra_blocks", "decoder_tra")
        self._fill_dict_thresholds(row, "dpgrnn1")
        self._fill_dict_thresholds(row, "dpgrnn2")
        return row

    def _fill_block_thresholds(self, row: dict, section: str, prefix: str):
        blocks = self.state.get(section, [])
        for idx, entry in enumerate(blocks):
            if entry is None:
                row[f"{prefix}_{idx}_x"] = None
                row[f"{prefix}_{idx}_h"] = None
            else:
                row[f"{prefix}_{idx}_x"] = entry.get("x")
                row[f"{prefix}_{idx}_h"] = entry.get("h")

    def _fill_dict_thresholds(self, row: dict, section: str):
        components = ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2")
        block = self.state.get(section, {})
        for name in components:
            values = block.get(name)
            if values is None:
                row[f"{section}_{name}_x"] = None
                row[f"{section}_{name}_h"] = None
            else:
                row[f"{section}_{name}_x"] = values.get("x")
                row[f"{section}_{name}_h"] = values.get("h")

    def _component_key(self, base: Path, component: str) -> str:
        stem = base.stem
        for prefix in self._component_prefixes:
            if stem.endswith(prefix):
                return f"{prefix}_{component}"
        return component

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
        if self.mode == "split":
            self._run_sweep_split()
            return
        if self.mode == "per_gru":
            self._run_sweep_per_gru()
            return
        rows = []
        occ_targets = []
        value = 0.0
        step = max(self.min_step, 1e-9)
        best_metric = -float("inf")
        best_state = None
        baseline_metric: Optional[float] = None
        start_time = time.time()
        iteration = 0
        while value <= self.max_threshold + 1e-9:
            iter_start = time.time()
            log_base = self.pickle_root / f"sweep_run_{self.run_counter}"
            self.run_counter += 1
            for param in self.params:
                param.setter(self.state, value)
            metric = self.evaluate(log_base=log_base)
            if baseline_metric is None:
                baseline_metric = metric
            row = {"threshold": value, "metric": metric}
            row.update(self._threshold_config_row())
            rows.append(row)
            occ_targets.append((row, log_base.parent / log_base.name))
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.state)
            if self.verbose:
                elapsed = time.time() - iter_start
                eta_text = ""
                if baseline_metric is not None and self.min_metric_drop is not None:
                    target = baseline_metric * (1.0 - self.min_metric_drop)
                    progress = baseline_metric - metric
                    total_needed = baseline_metric - target
                    if total_needed > 0:
                        fraction = min(max(progress / total_needed, 0.0), 1.0)
                        if fraction > 0:
                            total_elapsed = time.time() - start_time
                            eta = total_elapsed * (1 - fraction) / max(fraction, 1e-9)
                            eta_text = f", ETA ~ {eta/60:.1f} min"
                print(f"[opt] sweep value {value:.4f} -> metric {metric:.4f} (step took {elapsed:.1f}s{eta_text})")
            stop = False
            if metric < self.min_metric:
                stop = True
            if baseline_metric is not None and self.min_metric_drop is not None:
                if metric < baseline_metric * (1.0 - self.min_metric_drop):
                    stop = True
            if stop:
                break
            value += step
        self._write_csv(rows, occ_targets)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def _run_sweep_split(self):
        param_lookup = {param.name: param for param in self.params}
        if "global_x" not in param_lookup or "global_h" not in param_lookup:
            raise RuntimeError("Split mode sweep requires global_x and global_h parameters")
        x_param = param_lookup["global_x"]
        h_param = param_lookup["global_h"]

        rows = []
        occ_targets = []
        step = max(self.min_step, 1e-9)
        best_metric = -float("inf")
        best_state = None
        baseline_metric: Optional[float] = None
        start_time = time.time()

        h_value = 0.0
        stop_all = False
        initial_failures = 0
        while h_value <= self.max_threshold + 1e-9 and not stop_all:
            h_param.setter(self.state, h_value)
            x_value = 0.0
            first_iteration = True
            drop_violation = False
            if self.verbose and self.max_initial_failures > 0:
                print(
                    f"[opt] split sweep status before h={h_value:.4f}: "
                    f"consecutive drop-limit failures={initial_failures}/{self.max_initial_failures}"
                )
            while x_value <= self.max_threshold + 1e-9:
                iter_start = time.time()
                x_param.setter(self.state, x_value)
                log_base = self.pickle_root / f"sweep_run_{self.run_counter}"
                self.run_counter += 1
                metric = self.evaluate(log_base=log_base)
                if baseline_metric is None:
                    baseline_metric = metric
                row = {"threshold": x_value, "metric": metric}
                row.update(self._threshold_config_row())
                rows.append(row)
                occ_targets.append((row, log_base.parent / log_base.name))
                if metric > best_metric:
                    best_metric = metric
                    best_state = copy.deepcopy(self.state)
                if self.verbose:
                    elapsed = time.time() - iter_start
                    print(
                        f"[opt] sweep x={x_value:.4f}, h={h_value:.4f} -> metric {metric:.4f} "
                        f"(step took {elapsed:.1f}s)"
                    )
                stop_inner = False
                if metric < self.min_metric:
                    stop_inner = True
                    stop_all = True
                if (
                    not stop_inner
                    and baseline_metric is not None
                    and self.min_metric_drop is not None
                ):
                    drop_limit = baseline_metric * (1.0 - self.min_metric_drop)
                    if metric < drop_limit:
                        if first_iteration:
                            drop_violation = True
                            initial_failures += 1
                            if self.verbose:
                                print(
                                    f"[opt] split sweep drop-limit violation at h={h_value:.4f}; "
                                    f"consecutive failures={initial_failures}"
                                )
                            if self.max_initial_failures > 0 and initial_failures >= self.max_initial_failures:
                                stop_all = True
                        elif self.verbose:
                            total_elapsed = time.time() - start_time
                            print(
                                f"[opt] stopping x sweep at x={x_value:.4f}, h={h_value:.4f} "
                                f"after drop {baseline_metric - metric:.4f} (elapsed {total_elapsed/60:.1f} min)"
                            )
                        stop_inner = True
                if stop_inner:
                    break
                first_iteration = False
                x_value += step
            if stop_all:
                break
            h_value += step
            if not drop_violation:
                initial_failures = 0

        self._write_csv(rows, occ_targets)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def _run_sweep_per_gru(self):
        rows = []
        occ_targets = []
        step = max(self.min_step, 1e-9)
        best_metric = -float("inf")
        best_state = None
        baseline_metric: Optional[float] = None
        start_time = time.time()
        # Reset state so all thresholds start at zero for the per-GRU exploration
        self.state = build_initial_state(0.0, self.mode)

        stop_all = False
        for param in self.params:
            if stop_all:
                break
            value = 0.0
            while value <= self.max_threshold + 1e-9:
                iter_start = time.time()
                param.setter(self.state, value)
                log_base = self.pickle_root / f"sweep_run_{self.run_counter}"
                self.run_counter += 1
                metric = self.evaluate(log_base=log_base)
                if baseline_metric is None:
                    baseline_metric = metric
                row = {"threshold": value, "metric": metric}
                row.update(self._threshold_config_row())
                rows.append(row)
                occ_targets.append((row, log_base.parent / log_base.name))
                if metric > best_metric:
                    best_metric = metric
                    best_state = copy.deepcopy(self.state)
                if self.verbose:
                    elapsed = time.time() - iter_start
                    print(
                        f"[opt] sweep {param.name}={value:.4f} -> metric {metric:.4f} "
                        f"(step took {elapsed:.1f}s)"
                    )
                stop_inner = False
                if metric < self.min_metric:
                    stop_inner = True
                    stop_all = True
                if (
                    not stop_inner
                    and baseline_metric is not None
                    and self.min_metric_drop is not None
                ):
                    drop_limit = baseline_metric * (1.0 - self.min_metric_drop)
                    if metric < drop_limit:
                        if self.verbose:
                            total_elapsed = time.time() - start_time
                            print(
                                f"[opt] stopping sweep of {param.name} at {value:.4f} "
                                f"after drop {baseline_metric - metric:.4f} "
                                f"(elapsed {total_elapsed/60:.1f} min)"
                            )
                        stop_inner = True
                if stop_inner:
                    break
                value += step
            # Reset parameter before moving to the next one
            param.setter(self.state, 0.0)
        self._write_csv(rows, occ_targets)
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

    def _write_csv(self, rows: List[dict], occ_targets: List[Tuple[dict, Path]]):
        if not rows or self.csv_output is None:
            return
        fieldnames = ["threshold"] + self.threshold_columns + ["metric", "occ_x", "occ_y", "occ_tot"]
        for row, log_base in occ_targets:
            try:
                row.update(self.collect_occupancy(log_base))
            except FileNotFoundError:
                if self.verbose:
                    print(f"[opt] warning: occupancy logs missing for {log_base}")
            except Exception as exc:
                if self.verbose:
                    print(f"[opt] warning: failed occupancy aggregation for {log_base}: {exc}")
        self.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


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
    parser.add_argument("--max-initial-failures", type=int, default=0,
                        help="Stop split sweep when consecutive h values immediately violate min-metric")
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
        max_initial_failures=args.max_initial_failures,
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
