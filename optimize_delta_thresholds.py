#!/usr/bin/env python
import argparse
import copy
import csv
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from omegaconf import OmegaConf
import yaml

from delta_threshold_layout import (
    RUN_CSV_NAME,
    RUN_DIR_PREFIX,
    RUN_LOG_BASENAME,
    parse_run_index,
    per_component_columns,
)

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


def _set_group_threshold(state: dict, group: str, axis: str, value: Optional[float]):
    if axis not in ("x", "h"):
        raise ValueError(f"Unsupported axis '{axis}' (expected 'x' or 'h')")
    if group == "encoder":
        for block in state["encoder_tra_blocks"]:
            block[axis] = value
        return
    if group == "decoder":
        for block in state["decoder_tra_blocks"]:
            block[axis] = value
        return
    if group in ("dpgrnn1", "dpgrnn2"):
        for name in ("intra_rnn1", "intra_rnn2", "inter_rnn1", "inter_rnn2"):
            state[group][name][axis] = value
        return
    raise ValueError(f"Unsupported group '{group}' for threshold assignment")


def _get_group_threshold(state: dict, group: str, axis: str) -> Optional[float]:
    if axis not in ("x", "h"):
        raise ValueError(f"Unsupported axis '{axis}' (expected 'x' or 'h')")
    if group == "encoder":
        return state["encoder_tra_blocks"][0][axis]
    if group == "decoder":
        return state["decoder_tra_blocks"][0][axis]
    if group == "dpgrnn1":
        return state["dpgrnn1"]["intra_rnn1"][axis]
    if group == "dpgrnn2":
        return state["dpgrnn2"]["intra_rnn1"][axis]
    raise ValueError(f"Unsupported group '{group}' for threshold retrieval")


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
        if mode not in ("l1", "sensitivity"):
            raise ValueError(f"Unsupported mode {mode}")
        for group in ("encoder", "dpgrnn1", "dpgrnn2", "decoder"):
            params.append(ThresholdParam(
                f"{group}_x",
                getter=lambda s, group=group: _get_group_threshold(s, group, "x"),
                setter=lambda s, v, group=group: _set_group_threshold(s, group, "x", v),
            ))
            params.append(ThresholdParam(
                f"{group}_h",
                getter=lambda s, group=group: _get_group_threshold(s, group, "h"),
                setter=lambda s, v, group=group: _set_group_threshold(s, group, "h", v),
            ))
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
        csv_output: Optional[Path] = None,
        max_initial_failures: int = 0,
        resume_split: bool = False,
        resume_l1: bool = False,
        resume_sensitivity: bool = False,
        split_start_x: Optional[float] = None,
        split_start_h: Optional[float] = None,
        near_baseline_x: float = 0.0,
        near_baseline_h: float = 0.0,
        infer_max_files: Optional[int] = None,
        infer_no_copy: bool = False,
        infer_amp: bool = False,
        infer_workers: int = 1,
        sens_block: Optional[str] = None,
    ):
        self.infer_writer = ConfigWriter(infer_config)
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.pickle_root = self.work_dir / "pkls"
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
        self.csv_output = csv_output
        self.state = build_initial_state(0.0, mode)
        self.params = make_parameters(mode, 0.0)
        self.threshold_columns = self._build_threshold_columns()
        self.csv_fieldnames = ["threshold"] + self.threshold_columns + ["metric"]
        if self.mode == "sensitivity":
            self.csv_fieldnames = ["block"] + self.csv_fieldnames
        self.cache: Dict[Tuple, float] = {}
        self.best_metric: Optional[float] = None
        self.best_state: Optional[dict] = None
        self.max_initial_failures = max_initial_failures
        self.resume_split = resume_split
        self.resume_l1 = resume_l1
        self.resume_sensitivity = resume_sensitivity
        self.resume_info: Optional[dict] = None
        self.l1_resume_info: Optional[dict] = None
        self.sensitivity_resume_info: Optional[dict] = None
        self.sens_block = sens_block
        self.split_start_x = split_start_x
        self.split_start_h = split_start_h
        self.near_baseline_x = near_baseline_x
        self.near_baseline_h = near_baseline_h
        self.infer_max_files = infer_max_files
        self.infer_no_copy = infer_no_copy
        self.infer_amp = infer_amp
        self.infer_workers = infer_workers

        if (self.split_start_x is None) != (self.split_start_h is None):
            raise ValueError("split_start_x and split_start_h must both be provided")
        if self.mode != "split" and (self.split_start_x is not None or self.split_start_h is not None):
            raise ValueError("split start thresholds are only supported for mode 'split'")
        if self.resume_split and (self.split_start_x is not None or self.split_start_h is not None):
            raise ValueError("Cannot combine resume_split with explicit split start thresholds")

        if self.resume_split or self.resume_l1 or self.resume_sensitivity:
            self.pickle_root.mkdir(parents=True, exist_ok=True)
        else:
            if self.pickle_root.exists():
                shutil.rmtree(self.pickle_root)
            self.pickle_root.mkdir(parents=True, exist_ok=True)
        self.run_counter = 0
        if self.resume_split:
            self.resume_info = self._load_split_resume_info()
            baseline_state = self.resume_info["baseline_state"]
            self.state["global_x"] = baseline_state["global_x"]
            self.state["global_h"] = baseline_state["global_h"]
            self.run_counter = self.resume_info["next_run_index"]
        elif self.resume_l1:
            self.l1_resume_info = self._load_l1_resume_info()
            self.cache.update(self.l1_resume_info["cache"])
            self.best_metric = self.l1_resume_info["best_metric"]
            self.best_state = copy.deepcopy(self.l1_resume_info["best_state"]) if self.l1_resume_info["best_state"] is not None else None
            self.run_counter = self.l1_resume_info["next_run_index"]
            self.state = copy.deepcopy(self.l1_resume_info["resume_state"])
        elif self.resume_sensitivity:
            self.sensitivity_resume_info = self._load_sensitivity_resume_info()
            self.cache.update(self.sensitivity_resume_info["cache"])
            best_metric = self.sensitivity_resume_info["best_metric"]
            self.best_metric = best_metric if best_metric != -float("inf") else None
            self.best_state = copy.deepcopy(self.sensitivity_resume_info["best_state"]) if self.sensitivity_resume_info["best_state"] is not None else None
            self.run_counter = self.sensitivity_resume_info["next_run_index"]

    def evaluate(self, log_base: Optional[Path] = None) -> float:
        sig = state_signature(self.state)
        if sig in self.cache:
            return self.cache[sig]
        cfg_dict = build_thresholds_dict(self.state)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=self.work_dir, delete=False) as tmp:
            temp_path = Path(tmp.name)
            tmp.write(self.infer_writer.render(cfg_dict, log_file=log_base))
        try:
            infer_cmd = ["python", "infer.py", "-C", str(temp_path), "-D", self.device]
            if self.infer_no_copy:
                infer_cmd.append("--no-copy")
            if self.infer_amp:
                infer_cmd.append("--amp")
            if self.infer_max_files is not None:
                infer_cmd.extend(["--max-files", str(self.infer_max_files)])
            if self.infer_workers and self.infer_workers > 1:
                infer_cmd.extend(["--workers", str(self.infer_workers)])
            if self.verbose:
                print(f"[opt] running infer.py with {temp_path.name}")
            run_command(infer_cmd, verbose=self.verbose)
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
        columns = ["global", "global_x", "global_h"]
        columns.extend(per_component_columns())
        return columns

    def _threshold_config_row(self) -> Dict[str, Optional[float]]:
        row: Dict[str, Optional[float]] = {col: None for col in self.threshold_columns}
        if self.mode == "global":
            row["global"] = self.state["global_x"]
            return row
        if self.mode == "split":
            row["global_x"] = self.state["global_x"]
            row["global_h"] = self.state["global_h"]
            return row
        if self.mode not in ("per_gru", "l1", "sensitivity"):
            return row
        row["global_x"] = self.state["global_x"]
        row["global_h"] = self.state["global_h"]
        self._fill_block_thresholds(row, "encoder_tra_blocks", "enctra")
        self._fill_block_thresholds(row, "decoder_tra_blocks", "dectra")
        self._fill_dp_thresholds(row, "dpgrnn1", 1)
        self._fill_dp_thresholds(row, "dpgrnn2", 2)
        return row

    def _fill_block_thresholds(self, row: dict, section: str, prefix: str):
        blocks = self.state.get(section, [])
        for idx, entry in enumerate(blocks):
            base = f"{prefix}{idx}"
            if entry is None:
                row[f"{base}_x"] = None
                row[f"{base}_h"] = None
            else:
                row[f"{base}_x"] = entry.get("x")
                row[f"{base}_h"] = entry.get("h")

    def _fill_dp_thresholds(self, row: dict, section: str, dp_index: int):
        block = self.state.get(section, {})
        suffix_map = {
            "intra_rnn1": "intra1",
            "intra_rnn2": "intra2",
            "inter_rnn1": "inter1",
            "inter_rnn2": "inter2",
        }
        for name, suffix in suffix_map.items():
            values = block.get(name)
            base = f"dp{dp_index}{suffix}"
            if values is None:
                row[f"{base}_x"] = None
                row[f"{base}_h"] = None
            else:
                row[f"{base}_x"] = values.get("x")
                row[f"{base}_h"] = values.get("h")

    def _read_threshold_row(self, csv_path: Path) -> Dict[str, str]:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing thresholds CSV at {csv_path}")
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if row is None:
            raise RuntimeError(f"No rows found in {csv_path}")
        return row

    def _load_rows_from_runs(self) -> List[dict]:
        """Reconstruct sweep rows by reading the per-run CSVs in the work dir."""
        run_dirs = [p for p in self.pickle_root.glob(f"{RUN_DIR_PREFIX}*") if p.is_dir()]
        run_dirs = sorted(run_dirs, key=lambda p: parse_run_index(p)[0])
        rows: List[dict] = []
        for run_dir in run_dirs:
            index, _ = parse_run_index(run_dir)
            if index < 0:
                continue
            csv_path = run_dir / RUN_CSV_NAME
            if not csv_path.is_file():
                continue
            try:
                rows.append(self._read_threshold_row(csv_path))
            except RuntimeError:
                continue
        return rows

    @staticmethod
    def _parse_optional_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    return None
                return float(stripped)
            return float(value)
        except (TypeError, ValueError):
            return None

    def _state_from_row(self, row: Dict[str, str]) -> dict:
        state = build_initial_state(0.0, self.mode)
        gx = self._parse_optional_float(row.get("global_x"))
        gh = self._parse_optional_float(row.get("global_h"))
        fallback = self._parse_optional_float(row.get("global"))
        if gx is None:
            gx = fallback
        if gh is None:
            gh = fallback
        if gx is not None:
            state["global_x"] = gx
        if gh is not None:
            state["global_h"] = gh

        for key, raw in row.items():
            val = self._parse_optional_float(raw)
            if val is None:
                continue
            enc_match = re.match(r"enctra(\d+)_(x|h)$", key)
            if enc_match:
                idx, axis = enc_match.groups()
                enc_idx = int(idx)
                if 0 <= enc_idx < len(state["encoder_tra_blocks"]):
                    state["encoder_tra_blocks"][enc_idx][axis] = val
                continue
            dec_match = re.match(r"dectra(\d+)_(x|h)$", key)
            if dec_match:
                idx, axis = dec_match.groups()
                dec_idx = int(idx)
                if 0 <= dec_idx < len(state["decoder_tra_blocks"]):
                    state["decoder_tra_blocks"][dec_idx][axis] = val
                continue
            dp_match = re.match(r"dp(1|2)(intra|inter)(1|2)_(x|h)$", key)
            if dp_match:
                dp_idx, stage, branch, axis = dp_match.groups()
                target = state[f"dpgrnn{dp_idx}"]
                stage_key = f"{stage}_rnn{branch}"
                target[stage_key][axis] = val
                continue
        # Fill missing global_x/global_h using first non-None component value
        if gx is None:
            for section in ("encoder_tra_blocks", "decoder_tra_blocks"):
                for block in state[section]:
                    if block["x"] is not None:
                        gx = block["x"]
                        break
                if gx is not None:
                    break
            if gx is None:
                for dp_name in ("dpgrnn1", "dpgrnn2"):
                    for sub in state[dp_name].values():
                        if sub["x"] is not None:
                            gx = sub["x"]
                            break
                    if gx is not None:
                        break
        if gh is None:
            for section in ("encoder_tra_blocks", "decoder_tra_blocks"):
                for block in state[section]:
                    if block["h"] is not None:
                        gh = block["h"]
                        break
                if gh is not None:
                    break
            if gh is None:
                for dp_name in ("dpgrnn1", "dpgrnn2"):
                    for sub in state[dp_name].values():
                        if sub["h"] is not None:
                            gh = sub["h"]
                            break
                    if gh is not None:
                        break
        if gx is not None:
            state["global_x"] = gx
        if gh is not None:
            state["global_h"] = gh
        return state

    def _load_l1_resume_info(self) -> dict:
        if self.csv_output is None:
            raise RuntimeError("Cannot resume l1 near-search without --csv-output")
        if not self.csv_output.is_file():
            raise FileNotFoundError(f"Resume requested but summary CSV is missing at {self.csv_output}")
        with self.csv_output.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"Resume requested but no rows found in {self.csv_output}")

        cache: Dict[Tuple, float] = {}
        best_metric = -float("inf")
        best_state: Optional[dict] = None

        for row in rows:
            metric = self._parse_optional_float(row.get("metric"))
            state = self._state_from_row(row)
            sig = state_signature(state)
            if metric is not None:
                cache[sig] = metric
                if metric > best_metric:
                    best_metric = metric
                    best_state = copy.deepcopy(state)

        # Baseline from the first row of the sweep (original baseline run)
        baseline_state = self._state_from_row(rows[0])
        baseline_metric = self._parse_optional_float(rows[0].get("metric"))
        if baseline_metric is None:
            raise RuntimeError(f"Could not parse baseline metric from {self.csv_output}")

        run_dirs = [p for p in self.pickle_root.glob(f"{RUN_DIR_PREFIX}*") if p.is_dir()]
        run_dirs = sorted(run_dirs, key=lambda p: parse_run_index(p)[0])
        run_dirs = [p for p in run_dirs if parse_run_index(p)[0] >= 0 and (p / RUN_CSV_NAME).is_file()]
        if not run_dirs:
            raise RuntimeError(f"Resume requested but no run_* folders with {RUN_CSV_NAME} found under {self.pickle_root}")
        last_dir = run_dirs[-1]
        last_row = self._read_threshold_row(last_dir / RUN_CSV_NAME)
        resume_state = self._state_from_row(last_row)

        next_run_index = parse_run_index(last_dir)[0] + 1

        if best_metric == -float("inf"):
            best_metric = None

        return {
            "cache": cache,
            "best_metric": best_metric,
            "best_state": best_state,
            "baseline_metric": baseline_metric,
            "baseline_state": baseline_state,
            "rows": rows,
            "next_run_index": next_run_index,
            "resume_state": resume_state,
        }

    def _load_split_resume_info(self) -> dict:
        run_dirs = [p for p in self.pickle_root.glob(f"{RUN_DIR_PREFIX}*") if p.is_dir()]
        run_dirs = sorted(run_dirs, key=lambda p: parse_run_index(p)[0])
        run_dirs = [p for p in run_dirs if parse_run_index(p)[0] >= 0]
        if len(run_dirs) < 2:
            raise RuntimeError(
                "Resume requested but fewer than two run folders exist under "
                f"{self.pickle_root}"
            )
        prev_dir = run_dirs[-2]
        last_dir = run_dirs[-1]
        baseline_row = self._read_threshold_row(prev_dir / RUN_CSV_NAME)

        def _get_float(key: str) -> Optional[float]:
            raw = baseline_row.get(key)
            if raw is None or raw == "":
                return None
            try:
                return float(raw)
            except ValueError:
                return None

        baseline_x = _get_float("global_x") or _get_float("global")
        baseline_h = _get_float("global_h") or _get_float("global")
        if baseline_x is None or baseline_h is None:
            raise RuntimeError(
                f"Could not parse baseline x/h thresholds from {prev_dir / RUN_CSV_NAME}"
            )

        step = max(self.min_step, 1e-9)
        start_x = baseline_x + step
        if start_x > self.max_threshold + 1e-9:
            start_x = self.max_threshold
        last_index, _ = parse_run_index(last_dir)
        if last_index < 0:
            raise RuntimeError(f"Invalid run directory name: {last_dir.name}")
        if last_dir.exists():
            shutil.rmtree(last_dir)
        print(
            f"[opt] resuming split sweep from {prev_dir.name}: "
            f"h={baseline_h:.4f}, next x={start_x:.4f} (step {step:.4g}), "
            f"reusing index {last_index}"
        )
        return {
            "start_h": baseline_h,
            "start_x": start_x,
            "next_run_index": last_index,
            "baseline_state": {"global_x": baseline_x, "global_h": baseline_h},
        }

    def _load_sensitivity_resume_info(self) -> dict:
        rows = self._load_rows_from_runs()
        groups = self._sensitivity_groups()
        if self.sens_block:
            rows = [r for r in rows if r.get("block") == self.sens_block]
        loaded_from_csv = False
        if not rows and self.csv_output is not None and self.csv_output.is_file():
            with self.csv_output.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if self.sens_block:
                rows = [r for r in rows if r.get("block") == self.sens_block]
            loaded_from_csv = True
        if not rows:
            raise RuntimeError(
                "Resume requested but no completed sensitivity runs were found. "
                "Ensure the work directory contains run_* folders with thresholds.csv."
            )

        cache: Dict[Tuple, float] = {}
        best_metric = -float("inf")
        best_state: Optional[dict] = None

        for row in rows:
            metric = self._parse_optional_float(row.get("metric"))
            state = self._state_from_row(row)
            sig = state_signature(state)
            if metric is not None:
                cache[sig] = metric
                if metric > best_metric:
                    best_metric = metric
                    best_state = copy.deepcopy(state)

        run_dirs = [p for p in self.pickle_root.glob(f"{RUN_DIR_PREFIX}*") if p.is_dir()]
        run_dirs = [p for p in run_dirs if parse_run_index(p)[0] >= 0 and (p / RUN_CSV_NAME).is_file()]
        run_dirs = sorted(run_dirs, key=lambda p: parse_run_index(p)[0])
        next_run_index = len(rows)
        if run_dirs:
            last_dir = run_dirs[-1]
            index, _ = parse_run_index(last_dir)
            if index >= 0:
                next_run_index = max(next_run_index, index + 1)

        last_row = rows[-1]
        last_group = last_row.get("block")
        if last_group not in groups:
            source = str(self.csv_output) if loaded_from_csv and self.csv_output is not None else str(self.pickle_root)
            raise RuntimeError(f"Could not determine the last block from {source}")

        baseline_metric: Optional[float] = None
        for row in rows:
            if row.get("block") == last_group:
                baseline_metric = self._parse_optional_float(row.get("metric"))
                break
        if baseline_metric is None:
            source = str(self.csv_output) if loaded_from_csv and self.csv_output is not None else str(self.pickle_root)
            raise RuntimeError(f"Could not parse baseline metric for block '{last_group}' from {source}")

        last_state = self._state_from_row(last_row)
        last_x = _get_group_threshold(last_state, last_group, "x") or 0.0
        last_h = _get_group_threshold(last_state, last_group, "h") or 0.0
        last_metric = self._parse_optional_float(last_row.get("metric"))
        if last_metric is None:
            source = str(self.csv_output) if loaded_from_csv and self.csv_output is not None else str(self.pickle_root)
            raise RuntimeError(f"Last row in {source} is missing a metric value")

        step = max(self.min_step, 1e-9)
        next_group_index = groups.index(last_group)
        if self._metric_below_limit(last_metric, baseline_metric):
            next_x = 0.0
            next_h = last_h + step
        else:
            candidate_x = last_x + step
            if candidate_x <= self.max_threshold + 1e-9:
                next_x = candidate_x
                next_h = last_h
            else:
                next_x = 0.0
                next_h = last_h + step
        if next_h > self.max_threshold + 1e-9:
            next_group_index += 1
            next_x = 0.0
            next_h = 0.0
            baseline_metric = None
        if next_group_index > len(groups):
            next_group_index = len(groups)

        return {
            "rows": rows,
            "cache": cache,
            "best_metric": best_metric,
            "best_state": best_state,
            "next_run_index": next_run_index,
            "baseline_metric": baseline_metric,
            "next_group_index": next_group_index,
            "x_value": next_x,
            "h_value": next_h,
        }

    def _prepare_run_workspace(self) -> Tuple[Path, Path]:
        run_name = f"{RUN_DIR_PREFIX}{self.run_counter}"
        run_dir = self.pickle_root / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_base = run_dir / f"{RUN_LOG_BASENAME}_{run_name}"
        self.run_counter += 1
        return run_dir, log_base

    def _finalize_run_outputs(self, run_dir: Path, log_base: Path, row: dict):
        self._rename_run_pkls(run_dir, log_base)
        self._write_run_csv(run_dir, row)

    def _rename_run_pkls(self, run_dir: Path, log_base: Path):
        prefix = log_base.stem
        for path in list(run_dir.glob("*.pkl")):
            stem = path.stem
            if not stem.startswith(prefix):
                continue
            remainder = stem[len(prefix):]
            if remainder.startswith("_"):
                remainder = remainder[1:]
            new_stem = self._normalize_component_name(remainder)
            if not new_stem:
                continue
            dest = path.with_name(f"{new_stem}{path.suffix}")
            if dest.exists():
                dest.unlink()
            path.rename(dest)
        raw_base = log_base.with_suffix(".pkl")
        if raw_base.exists():
            raw_base.unlink()

    def _normalize_component_name(self, name: str) -> Optional[str]:
        name = name.lstrip("_")
        if not name:
            return None
        match = re.match(r"enc_tra(\d+)_(x|h)$", name)
        if match:
            idx, axis = match.groups()
            return f"enctra{idx}_{axis}"
        match = re.match(r"dec_tra(\d+)_(x|h)$", name)
        if match:
            idx, axis = match.groups()
            return f"dectra{idx}_{axis}"
        match = re.match(r"dp(\d+)_(intra|inter)_(x|h)([12])$", name)
        if match:
            dp_idx, stage, axis, branch = match.groups()
            return f"dp{dp_idx}{stage}{branch}_{axis}"
        return name

    def _sensitivity_groups(self) -> Tuple[str, ...]:
        if self.sens_block:
            return (self.sens_block,)
        return ("encoder", "dpgrnn1", "dpgrnn2", "decoder")

    def _write_run_csv(self, run_dir: Path, row: dict):
        csv_path = run_dir / RUN_CSV_NAME
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            writer.writerow(row)

    def _write_summary_csv(self, rows: List[dict]):
        if not rows or self.csv_output is None:
            return
        self.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _apply_baseline_thresholds(self, x_value: float, h_value: float):
        self.state["global_x"] = x_value
        self.state["global_h"] = h_value
        for group in ("encoder", "dpgrnn1", "dpgrnn2", "decoder"):
            _set_group_threshold(self.state, group, "x", x_value)
            _set_group_threshold(self.state, group, "h", h_value)

    def _metric_below_limit(self, metric: float, baseline_metric: Optional[float]) -> bool:
        if metric < self.min_metric:
            return True
        if baseline_metric is not None and self.min_metric_drop is not None:
            return metric < baseline_metric * (1.0 - self.min_metric_drop)
        return False

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
        value = 0.0
        step = max(self.min_step, 1e-9)
        best_metric = -float("inf")
        best_state = None
        baseline_metric: Optional[float] = None
        start_time = time.time()
        while value <= self.max_threshold + 1e-9:
            iter_start = time.time()
            run_dir, log_base = self._prepare_run_workspace()
            for param in self.params:
                param.setter(self.state, value)
            metric = self.evaluate(log_base=log_base)
            if baseline_metric is None:
                baseline_metric = metric
            row = {"threshold": value, "metric": metric}
            row.update(self._threshold_config_row())
            rows.append(row)
            self._finalize_run_outputs(run_dir, log_base, row)
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
        self._write_summary_csv(rows)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def run_near_search(self):
        if self.mode != "l1":
            raise RuntimeError("near-search strategy is only supported for mode 'l1'")
        step = max(self.min_step, 1e-9)
        rows: List[dict] = []
        baseline_metric: Optional[float] = None
        initial_state: Optional[dict] = None

        if self.resume_l1 and self.l1_resume_info is not None:
            rows = [dict(r) for r in self.l1_resume_info["rows"]]
            baseline_metric = self.l1_resume_info["baseline_metric"]
            self.state = copy.deepcopy(self.l1_resume_info["resume_state"])
            baseline_state = self.l1_resume_info["baseline_state"]
            # Warn if CLI baseline differs from saved baseline, but keep CLI values
            saved_x = baseline_state.get("global_x")
            saved_h = baseline_state.get("global_h")
            if saved_x is not None and abs(saved_x - self.near_baseline_x) > 1e-9:
                print(
                    f"[opt] warning: saved baseline x={saved_x} differs from --near-baseline-x={self.near_baseline_x}; "
                    f"using CLI value"
                )
            if saved_h is not None and abs(saved_h - self.near_baseline_h) > 1e-9:
                print(
                    f"[opt] warning: saved baseline h={saved_h} differs from --near-baseline-h={self.near_baseline_h}; "
                    f"using CLI value"
                )
            if self.best_metric is None:
                self.best_metric = self.l1_resume_info["best_metric"]
            if self.best_state is None and self.l1_resume_info["best_state"] is not None:
                self.best_state = copy.deepcopy(self.l1_resume_info["best_state"])
            initial_state = copy.deepcopy(self.state)
        else:
            # Start from the user-provided baseline applied to all groups
            self._apply_baseline_thresholds(self.near_baseline_x, self.near_baseline_h)
            run_dir, log_base = self._prepare_run_workspace()
            baseline_metric = self.evaluate(log_base=log_base)
            baseline_row = {"threshold": 0.0, "metric": baseline_metric}
            baseline_row.update(self._threshold_config_row())
            rows.append(baseline_row)
            self._finalize_run_outputs(run_dir, log_base, baseline_row)
            self.best_metric = baseline_metric
            self.best_state = copy.deepcopy(self.state)
            initial_state = copy.deepcopy(self.state)

        if baseline_metric is None:
            raise RuntimeError("Baseline metric is missing; cannot continue near-search")

        def _update_best(metric: float):
            if self.best_metric is None or metric > self.best_metric:
                self.best_metric = metric
                self.best_state = copy.deepcopy(self.state)

        groups = self._sensitivity_groups()

        def _metric_ok(metric: float) -> bool:
            return not self._metric_below_limit(metric, baseline_metric)

        def _start_value_for(group: str, axis: str) -> float:
            if initial_state is None:
                return self.near_baseline_x if axis == "x" else self.near_baseline_h
            current = _get_group_threshold(initial_state, group, axis)
            if current is None:
                return self.near_baseline_x if axis == "x" else self.near_baseline_h
            return current

        def _evaluate_and_record(tag_value: float) -> bool:
            iter_run_dir, iter_log_base = self._prepare_run_workspace()
            metric = self.evaluate(log_base=iter_log_base)
            row = {"threshold": tag_value, "metric": metric}
            row.update(self._threshold_config_row())
            rows.append(row)
            self._finalize_run_outputs(iter_run_dir, iter_log_base, row)
            _update_best(metric)
            return _metric_ok(metric)

        def _reset_inner(end_index: int):
            for idx in range(end_index):
                inner_group = groups[idx]
                _set_group_threshold(self.state, inner_group, "x", _start_value_for(inner_group, "x"))
                _set_group_threshold(self.state, inner_group, "h", _start_value_for(inner_group, "h"))

        def _optimize_chain(level: int) -> bool:
            """Optimize groups[0..level] with nested x-then-h sweeps, returning True if any step succeeded."""
            group = groups[level]
            has_success = False
            h_value = _start_value_for(group, "h")
            last_good_h = h_value

            while True:  # Sweep x for the current h, then try to bump h
                x_value = _start_value_for(group, "x")
                last_good_x = x_value
                while True:
                    candidate_x = x_value + step
                    if candidate_x > self.max_threshold + 1e-9:
                        break
                    _set_group_threshold(self.state, group, "x", candidate_x)
                    _set_group_threshold(self.state, group, "h", h_value)
                    _reset_inner(level)
                    if level > 0:
                        success = _optimize_chain(level - 1)
                    else:
                        success = _evaluate_and_record(candidate_x)
                    if not success:
                        _set_group_threshold(self.state, group, "x", last_good_x)
                        break
                    has_success = True
                    x_value = candidate_x
                    last_good_x = x_value

                candidate_h = h_value + step
                if candidate_h > self.max_threshold + 1e-9:
                    _set_group_threshold(self.state, group, "h", last_good_h)
                    _set_group_threshold(self.state, group, "x", last_good_x)
                    break
                _set_group_threshold(self.state, group, "h", candidate_h)
                _set_group_threshold(self.state, group, "x", self.near_baseline_x)
                _reset_inner(level)
                if level > 0:
                    success = _optimize_chain(level - 1)
                else:
                    success = _evaluate_and_record(candidate_h)
                if not success:
                    _set_group_threshold(self.state, group, "h", last_good_h)
                    _set_group_threshold(self.state, group, "x", last_good_x)
                    break
                has_success = True
                h_value = candidate_h
                last_good_h = h_value

            return has_success

        # Run the cascaded search: start with the innermost group, then grow outward
        max_level = len(groups) - 1
        for level in range(max_level + 1):
            _optimize_chain(level)

        self._write_summary_csv(rows)
        if self.apply_best_to is not None and self.best_state is not None:
            cfg_dict = build_thresholds_dict(self.best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def _run_sweep_split(self):
        param_lookup = {param.name: param for param in self.params}
        if "global_x" not in param_lookup or "global_h" not in param_lookup:
            raise RuntimeError("Split mode sweep requires global_x and global_h parameters")
        x_param = param_lookup["global_x"]
        h_param = param_lookup["global_h"]

        rows = []
        step = max(self.min_step, 1e-9)
        best_metric = -float("inf")
        best_state = None
        baseline_metric: Optional[float] = None
        start_time = time.time()

        if self.resume_info:
            h_value = self.resume_info["start_h"]
        else:
            h_value = self.split_start_h if self.split_start_h is not None else 0.0
        stop_all = False
        initial_failures = 0
        resume_active = self.resume_info is not None
        first_h_iteration = True
        while h_value <= self.max_threshold + 1e-9 and not stop_all:
            h_param.setter(self.state, h_value)
            if resume_active and first_h_iteration:
                x_value = self.resume_info["start_x"]
            else:
                x_value = self.split_start_x if self.split_start_x is not None else 0.0
            first_iteration = True
            drop_violation = False
            if self.verbose and self.max_initial_failures > 0:
                print(
                    f"[opt] split sweep status before h={h_value:.4f}: "
                    f"consecutive drop-limit failures={initial_failures}/{self.max_initial_failures}"
                )
            while x_value <= self.max_threshold + 1e-9:
                iter_start = time.time()
                run_dir, log_base = self._prepare_run_workspace()
                x_param.setter(self.state, x_value)
                metric = self.evaluate(log_base=log_base)
                if baseline_metric is None:
                    baseline_metric = metric
                row = {"threshold": x_value, "metric": metric}
                row.update(self._threshold_config_row())
                rows.append(row)
                self._finalize_run_outputs(run_dir, log_base, row)
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
                min_floor_violation = metric < self.min_metric
                if min_floor_violation:
                    stop_inner = True
                    if self.verbose:
                        print(
                            f"[opt] metric {metric:.4f} fell below min-metric {self.min_metric:.4f} "
                            f"at x={x_value:.4f}, h={h_value:.4f}; advancing h and resetting x"
                        )
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
            resume_active = False
            first_h_iteration = False

        self._write_summary_csv(rows)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def _run_sweep_per_gru(self):
        rows = []
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
                run_dir, log_base = self._prepare_run_workspace()
                param.setter(self.state, value)
                metric = self.evaluate(log_base=log_base)
                if baseline_metric is None:
                    baseline_metric = metric
                row = {"threshold": value, "metric": metric}
                row.update(self._threshold_config_row())
                rows.append(row)
                self._finalize_run_outputs(run_dir, log_base, row)
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
        self._write_summary_csv(rows)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def run_sensitivity(self):
        if self.mode != "sensitivity":
            raise RuntimeError("Sensitivity mode is only available with --mode sensitivity")
        step = max(self.min_step, 1e-9)
        rows: List[dict] = []
        best_metric = -float("inf")
        best_state: Optional[dict] = None
        groups = self._sensitivity_groups()
        start_time = time.time()

        start_group_index = 0
        resume_h = 0.0
        resume_x = 0.0
        resume_baseline: Optional[float] = None
        resume_active = False
        if self.resume_sensitivity and self.sensitivity_resume_info is not None:
            rows = [dict(r) for r in self.sensitivity_resume_info["rows"]]
            cached_best = self.sensitivity_resume_info["best_metric"]
            if cached_best is not None and cached_best != -float("inf"):
                best_metric = cached_best
            best_state = copy.deepcopy(self.sensitivity_resume_info["best_state"]) if self.sensitivity_resume_info["best_state"] is not None else None
            start_group_index = self.sensitivity_resume_info["next_group_index"]
            resume_h = self.sensitivity_resume_info["h_value"]
            resume_x = self.sensitivity_resume_info["x_value"]
            resume_baseline = self.sensitivity_resume_info["baseline_metric"]
            resume_active = start_group_index < len(groups)

        for idx, group in enumerate(groups):
            if idx < start_group_index:
                continue
            # Reset thresholds so only the current group is swept; others stay at zero
            self.state = build_initial_state(0.0, self.mode)
            for reset_group in groups:
                _set_group_threshold(self.state, reset_group, "x", 0.0)
                _set_group_threshold(self.state, reset_group, "h", 0.0)

            baseline_metric: Optional[float] = resume_baseline if resume_active and idx == start_group_index else None
            h_value = resume_h if resume_active and idx == start_group_index else 0.0
            first_h_iteration = True
            while h_value <= self.max_threshold + 1e-9:
                if resume_active and idx == start_group_index and first_h_iteration:
                    x_value = resume_x
                else:
                    x_value = 0.0
                first_h_iteration = False
                while x_value <= self.max_threshold + 1e-9:
                    iter_start = time.time()
                    run_dir, log_base = self._prepare_run_workspace()
                    _set_group_threshold(self.state, group, "x", x_value)
                    _set_group_threshold(self.state, group, "h", h_value)
                    metric = self.evaluate(log_base=log_base)
                    if baseline_metric is None:
                        baseline_metric = metric
                    row = {"block": group, "threshold": x_value, "metric": metric}
                    row.update(self._threshold_config_row())
                    rows.append(row)
                    self._finalize_run_outputs(run_dir, log_base, row)
                    if metric > best_metric:
                        best_metric = metric
                        best_state = copy.deepcopy(self.state)
                    if self.verbose:
                        elapsed = time.time() - iter_start
                        elapsed_total = (time.time() - start_time) / 60
                        print(
                            f"[opt] sensitivity sweep {group}: x={x_value:.4f}, h={h_value:.4f} "
                            f"-> metric {metric:.4f} (step {elapsed:.1f}s, total {elapsed_total:.1f}m)"
                        )
                    if self._metric_below_limit(metric, baseline_metric):
                        break
                    x_value += step
                h_value += step
            resume_active = False

        self._write_summary_csv(rows)
        self.best_metric = best_metric if best_metric != -float("inf") else None
        self.best_state = best_state
        if self.apply_best_to is not None and best_state is not None:
            cfg_dict = build_thresholds_dict(best_state)
            writer = ConfigWriter(self.apply_best_to)
            writer.write_to(self.apply_best_to, cfg_dict)

    def run(self):
        if self.mode == "sensitivity":
            self.run_sensitivity()
            return
        if self.mode == "l1" and self.strategy != "near-search":
            raise RuntimeError("Mode 'l1' requires strategy 'near-search'")
        if self.strategy == "coordinate":
            self.run_coordinate()
        elif self.strategy == "sweep":
            self.run_sweep()
        elif self.strategy == "near-search":
            self.run_near_search()
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")


def main():
    parser = argparse.ArgumentParser(description="Optimize DeltaGRU thresholds via divide-and-conquer or sweep search")
    parser.add_argument("--infer-config", default="configs/cfg_infer.yaml", help="Inference config path")
    parser.add_argument("--train-config", default=None, help="Optional cfg_train.yaml to update with best thresholds")
    parser.add_argument("--mode", choices=["global", "split", "per_gru", "l1", "sensitivity"], default="global")
    parser.add_argument("--strategy", choices=["coordinate", "sweep", "near-search"], default="coordinate")
    parser.add_argument("--metric", choices=list(RESULT_METRICS), default="PESQ")
    parser.add_argument("--max-threshold", type=float, default=1.0, help="Upper bound for thresholds")
    parser.add_argument("--decay", type=float, default=0.5, help="Step decay factor (e.g. 0.5 for halving)")
    parser.add_argument(
        "--min-step",
        type=float,
        default=0.05,
        help="Smallest step size before stopping; also used as the increment for near-search",
    )
    parser.add_argument("--work-dir", default="logs/threshold_opt", help="Working directory for temp configs")
    parser.add_argument("--device", default="0", help="GPU device id for infer.py")
    parser.add_argument(
        "--min-metric",
        type=float,
        default=0.0,
        help=(
            "Absolute metric floor. In split sweeps, falling below this value skips to the next h "
            "(resetting x); in other sweeps it stops the search."
        ),
    )
    parser.add_argument("--max-metric-drop", type=float, default=None, help="Relative drop (e.g. 0.15 for 15%) allowed vs baseline")
    parser.add_argument("--csv-output", default="logs/threshold_opt/sweep_results.csv", help="CSV file for sweep summaries")
    parser.add_argument("--max-initial-failures", type=int, default=0,
                        help="Stop split sweep when consecutive h values immediately violate min-metric")
    parser.add_argument("--resume-split", action="store_true",
                        help="Resume split sweep: restart from the last completed (run_*) thresholds and continue with x+=step")
    parser.add_argument("--resume-l1", action="store_true",
                        help="Resume l1 near-search: reuse prior runs and summary CSV to continue search")
    parser.add_argument("--resume-sensitivity", action="store_true",
                        help="Resume sensitivity sweep using the existing summary CSV and run folders")
    parser.add_argument("--sens-block", choices=["encoder", "decoder", "dpgrnn1", "dpgrnn2"], default=None,
                        help="When in sensitivity mode, restrict the sweep to a single block")
    parser.add_argument("--split-x", type=float, default=None,
                        help="Starting x threshold for split sweeps (mode=split, strategy=sweep)")
    parser.add_argument("--split-h", type=float, default=None,
                        help="Starting h threshold for split sweeps (mode=split, strategy=sweep)")
    parser.add_argument("--near-baseline-x", type=float, default=0.0,
                        help="Baseline x threshold used to seed near-search (applied to all groups)")
    parser.add_argument("--near-baseline-h", type=float, default=0.0,
                        help="Baseline h threshold used to seed near-search (applied to all groups)")
    parser.add_argument("--infer-max-files", type=int, default=None,
                        help="Limit inference to the first N files (speeds sweeps; affects metrics).")
    parser.add_argument("--infer-no-copy", action="store_true",
                        help="Skip copying noisy/clean wavs into enh_folder to cut I/O.")
    parser.add_argument("--infer-amp", action="store_true",
                        help="Enable mixed precision in infer.py (CUDA only).")
    parser.add_argument("--infer-workers", type=int, default=1,
                        help="Number of parallel infer.py workers (each loads its own model).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.resume_split and (args.strategy != "sweep" or args.mode != "split"):
        parser.error("--resume-split is only supported for split sweeps (strategy=sweep, mode=split)")
    if args.strategy == "near-search" and args.mode != "l1":
        parser.error("near-search strategy is only supported with --mode l1")
    if args.mode == "l1" and args.strategy != "near-search":
        parser.error("mode l1 requires --strategy near-search")
    if args.resume_l1 and (args.mode != "l1" or args.strategy != "near-search"):
        parser.error("--resume-l1 is only supported for l1 near-search")
    if args.resume_l1 and args.resume_split:
        parser.error("Cannot combine --resume-l1 with --resume-split")
    if args.resume_sensitivity and args.mode != "sensitivity":
        parser.error("--resume-sensitivity is only supported for sensitivity sweeps (mode=sensitivity)")
    if args.resume_sensitivity and (args.resume_l1 or args.resume_split):
        parser.error("Cannot combine --resume-sensitivity with other resume modes")
    if args.sens_block and args.mode != "sensitivity":
        parser.error("--sens-block is only valid in sensitivity mode")
    if (args.split_x is None) != (args.split_h is None):
        parser.error("--split-x and --split-h must both be provided")
    if args.split_x is not None:
        if args.mode != "split" or args.strategy != "sweep":
            parser.error("--split-x/--split-h are only supported for split sweeps (strategy=sweep, mode=split)")
        if args.resume_split:
            parser.error("Cannot combine --split-x/--split-h with --resume-split")

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
        csv_output=Path(args.csv_output).resolve() if args.csv_output else None,
        max_initial_failures=args.max_initial_failures,
        resume_split=args.resume_split,
        resume_l1=args.resume_l1,
        resume_sensitivity=args.resume_sensitivity,
        split_start_x=args.split_x,
        split_start_h=args.split_h,
        near_baseline_x=args.near_baseline_x,
        near_baseline_h=args.near_baseline_h,
        infer_max_files=args.infer_max_files,
        infer_no_copy=args.infer_no_copy,
        infer_amp=args.infer_amp,
        infer_workers=args.infer_workers,
        sens_block=args.sens_block,
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
