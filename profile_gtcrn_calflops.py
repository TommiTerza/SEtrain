#!/usr/bin/env python3
"""Profile the core GTCRN network (without FFT) using calflops."""
import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.gtcrn_end2end import GTCRN, GTCRNCore


def _format(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3f} G"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f} M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f} K"
    return f"{value:.3f}"



_CONV_LAYERS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

_NORM_LAYERS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

_ACTIVATION_LAYERS = (
    nn.ReLU,
    nn.ReLU6,
    nn.PReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    nn.Softplus,
    nn.Sigmoid,
    nn.Tanh,
    nn.SiLU,
    nn.Hardtanh,
)


def _is_activation(module: nn.Module) -> bool:
    return isinstance(module, _ACTIVATION_LAYERS)


def _is_normalization(module: nn.Module) -> bool:
    return isinstance(module, _NORM_LAYERS)


def _is_convolution(module: nn.Module) -> bool:
    return isinstance(module, _CONV_LAYERS)


def _infer_block(name: str) -> str:
    return name.split(".", 1)[0]


def _block_component_category(block: str, name: str, module: nn.Module) -> str:
    if ".tra." in name or name.endswith(".tra"):
        return "TRA"
    if "sfe" in name and module.__class__.__name__ == "SFE":
        return "SFE"
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return "BatchNorm"
    if isinstance(module, nn.LayerNorm):
        return "LayerNorm"
    if _is_convolution(module):
        if "point_conv" in name:
            return "PointConv"
        if "depth_conv" in name:
            return "DepthConv"
        return "Conv"
    if isinstance(module, nn.GRU):
        return "GRU"
    if isinstance(module, nn.Linear):
        return "Linear"
    if _is_activation(module):
        return "Activation"
    if _is_normalization(module):
        return "Normalization"
    if module.__class__.__name__ == "SFE":
        return "SFE"
    return "Other"


def _operation_family(name: str, module: nn.Module) -> str:
    class_name = module.__class__.__name__
    if class_name == "SFE":
        return "SFE"
    if _is_convolution(module):
        return "Convolution"
    if isinstance(module, nn.GRU):
        return "GRU"
    if isinstance(module, nn.Linear):
        return "Linear"
    if _is_normalization(module):
        return "Normalization"
    if _is_activation(module):
        return "Activation"
    if class_name.lower().startswith("dropout"):
        return "Dropout"
    return "Other"


def _profile_latency(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    warmup: int,
    runs: int,
    top_k: int,
) -> None:
    if runs <= 0:
        print("\nLatency profiling skipped (latency runs <= 0).")
        return

    dummy_input = torch.randn(*input_shape, device=device)

    module_lookup: Dict[str, nn.Module] = {}
    is_leaf_map: Dict[str, bool] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        module_lookup[name] = module
        is_leaf_map[name] = not any(module.children())

    if not module_lookup:
        print("\nLatency profiling skipped (no measurable modules).")
        return

    latencies: Dict[str, List[float]] = defaultdict(list)
    start_times: Dict[str, float] = {}

    def make_pre_hook(name: str):
        def _pre_hook(*_):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_times[name] = time.perf_counter()

        return _pre_hook

    def make_post_hook(name: str):
        def _post_hook(*_):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = start_times.pop(name, None)
            if start is None:
                return
            latencies[name].append(time.perf_counter() - start)

        return _post_hook

    handles = []
    totals: List[float] = []
    try:
        for name, module in module_lookup.items():
            handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
            handles.append(module.register_forward_hook(make_post_hook(name)))

        with torch.inference_mode():
            for _ in range(max(warmup, 0)):
                model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        with torch.inference_mode():
            for _ in range(runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_start = time.perf_counter()
                model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                totals.append(time.perf_counter() - total_start)
    finally:
        for handle in handles:
            handle.remove()

    if not totals:
        print("\nLatency profiling skipped (no runs executed).")
        return

    total_mean = statistics.mean(totals)
    total_median = statistics.median(totals)
    total_std = statistics.pstdev(totals) if len(totals) > 1 else 0.0

    print("\n=== Latency profile (GTCRN core) ===")
    print(f"device: {device}")
    print(f"warmup runs: {warmup}")
    print(f"measured runs: {runs}")
    print(f"Average latency: {total_mean * 1e3:.3f} ms")
    print(f"Median latency: {total_median * 1e3:.3f} ms")
    print(f"Latency std-dev: {total_std * 1e3:.3f} ms")

    module_summary: Dict[str, Dict[str, object]] = {}
    group_totals: Dict[str, float] = defaultdict(float)
    group_calls: Dict[str, float] = defaultdict(float)
    module_stats = []

    for name, durations in latencies.items():
        module = module_lookup.get(name)
        if module is None or not durations:
            continue
        total_time = sum(durations)
        calls = len(durations)
        per_run = total_time / runs
        per_call = total_time / calls
        std = statistics.pstdev(durations) if calls > 1 else 0.0
        share = (per_run / total_mean * 100.0) if total_mean > 0 else 0.0

        info = {
            "per_run": per_run,
            "per_call": per_call,
            "std": std,
            "calls": calls,
            "share": share,
            "module": module,
            "is_leaf": is_leaf_map.get(name, False),
        }
        module_summary[name] = info

        if info["is_leaf"]:
            group = _infer_block(name)
            group_totals[group] += total_time
            group_calls[group] += calls
            module_stats.append((name, per_run, per_call, std, calls, share))

    if not module_summary:
        print("\nLatency profiling skipped (no module activations recorded).")
        return

    if module_stats:
        print("\nTop-level modules (per-sample mean):")
        header = f"{'module':30s} {'time (ms)':>12s} {'share (%)':>12s} {'calls/run':>12s}"
        print(header)
        print("-" * len(header))
        for group, total_time in sorted(group_totals.items(), key=lambda item: item[1], reverse=True):
            per_run = total_time / runs
            share = (per_run / total_mean * 100.0) if total_mean > 0 else 0.0
            calls_per_run = group_calls[group] / runs
            print(f"{group:30s} {per_run * 1e3:12.3f} {share:12.2f} {calls_per_run:12.2f}")

        if top_k != 0:
            limit = top_k if top_k > 0 else len(module_stats)
            print(f"\nMost expensive modules (top {limit} by per-sample time):")
            header = (
                f"{'module':50s} {'time/run (ms)':>14s} {'time/call (ms)':>15s} "
                f"{'std (ms)':>10s} {'calls/run':>12s} {'share (%)':>10s}"
            )
            print(header)
            print("-" * len(header))
            for name, per_run, per_call, std, calls, share in sorted(
                module_stats, key=lambda item: item[1], reverse=True
            )[:limit]:
                calls_per_run = calls / runs
                print(
                    f"{name:50s} {per_run * 1e3:14.3f} {per_call * 1e3:15.3f} "
                    f"{std * 1e3:10.3f} {calls_per_run:12.2f} {share:10.2f}"
                )

    leaf_infos = {
        name: info for name, info in module_summary.items() if info["is_leaf"]
    }
    total_leaf_time = sum(info["per_run"] for info in leaf_infos.values())

    block_order = ("erb", "encoder", "dpgrnn1", "dpgrnn2", "decoder")
    block_components: Dict[str, Dict[str, float]] = {
        block: defaultdict(float) for block in block_order
    }
    block_totals = defaultdict(float)

    for name, info in leaf_infos.items():
        per_run = info["per_run"]
        if per_run <= 0:
            continue
        block = _infer_block(name)
        if block not in block_components:
            continue
        module = info["module"]
        category = _block_component_category(block, name, module)
        block_components[block][category] += per_run
        block_totals[block] += per_run

    dpgrnn_combined = defaultdict(float)
    for block in ("dpgrnn1", "dpgrnn2"):
        for category, value in block_components.get(block, {}).items():
            dpgrnn_combined[category] += value
    block_components["dpgrnn"] = dpgrnn_combined
    block_totals["dpgrnn"] = block_totals["dpgrnn1"] + block_totals["dpgrnn2"]

    print("\nMain blocks (per-sample mean, leaf modules):")
    header = f"{'block':20s} {'time (ms)':>12s} {'share (%)':>12s}"
    print(header)
    print("-" * len(header))
    for block in ("erb", "encoder", "dpgrnn1", "dpgrnn2", "dpgrnn", "decoder"):
        per_run = block_totals.get(block, 0.0)
        share = (per_run / total_mean * 100.0) if total_mean > 0 else 0.0
        print(f"{block:20s} {per_run * 1e3:12.3f} {share:12.2f}")

    for block in ("encoder", "dpgrnn1", "dpgrnn2", "dpgrnn", "decoder", "erb"):
        components = block_components.get(block, {})
        if not components:
            continue
        block_total = block_totals.get(block, 0.0)
        print(f"\nBlock '{block}' breakdown (per-sample mean):")
        header = f"{'component':25s} {'time (ms)':>12s} {'share (%)':>12s}"
        print(header)
        print("-" * len(header))
        for component, value in sorted(components.items(), key=lambda item: item[1], reverse=True):
            share = (value / block_total * 100.0) if block_total > 0 else 0.0
            print(f"{component:25s} {value * 1e3:12.3f} {share:12.2f}")

    type_totals: Dict[str, float] = defaultdict(float)
    for name, info in leaf_infos.items():
        per_run = info["per_run"]
        if per_run <= 0:
            continue
        module = info["module"]
        family = _operation_family(name, module)
        type_totals[family] += per_run

    accounted = sum(type_totals.values())
    overhead = max(total_mean - accounted, 0.0)
    if overhead > 0:
        type_totals["Overhead"] += overhead

    print("\nLatency by operation type (per-sample mean):")
    header = f"{'type':20s} {'time (ms)':>12s} {'share (%)':>12s}"
    print(header)
    print("-" * len(header))
    for op_type, value in sorted(type_totals.items(), key=lambda item: item[1], reverse=True):
        share = (value / total_mean * 100.0) if total_mean > 0 else 0.0
        print(f"{op_type:20s} {value * 1e3:12.3f} {share:12.2f}")

    if total_leaf_time < total_mean:
        gap = total_mean - total_leaf_time
        print(
            f"\nNote: {gap * 1e3:.3f} ms ({(gap / total_mean * 100.0) if total_mean > 0 else 0.0:.2f}%) "
            "of latency comes from operations outside measured modules (e.g., tensor reshapes)."
        )

def _resolve_device(use_cuda: bool, device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA device requested but not available.")
        return torch.device("cuda")

    # auto mode keeps backward compatible --cuda flag
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if use_cuda and not torch.cuda.is_available():
        print("Warning: --cuda supplied but CUDA is unavailable; falling back to CPU.")
    return torch.device("cpu")


def profile_model(
    config: Path,
    batch_size: int,
    frames: int,
    use_cuda: bool,
    print_detailed: bool,
    latency: bool = False,
    latency_warmup: int = 5,
    latency_runs: int = 20,
    latency_top_k: int = 20,
    device_pref: str = "auto",
) -> None:
    try:
        from calflops import calculate_flops  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        missing = exc.name or "calflops"
        if missing != "calflops":
            msg = (
                f"Dependency '{missing}' required by calflops is missing. "
                "Install it in the current environment and rerun."
            )
        else:
            msg = "calflops is not installed. Install it with `pip install calflops` and rerun."
        raise SystemExit(msg) from exc
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Failed to import calflops. Ensure it is installed in the active environment.") from exc

    cfg = OmegaConf.load(str(config))
    model_kwargs = dict(cfg.network_config)

    device = _resolve_device(use_cuda, device_pref)
    full_model = GTCRN(**model_kwargs).to(device)
    full_model.eval()

    core_model = GTCRNCore.from_full_model(full_model)
    core_model.eval()

    freq_bins = full_model.n_fft // 2 + 1
    input_shape = (batch_size, 3, frames, freq_bins)

    flops, macs, params = calculate_flops(
        model=core_model,
        input_shape=input_shape,
        print_results=print_detailed,
        print_detailed=print_detailed,
        output_as_string=False,
    )

    print("\n=== calflops profile (GTCRN core) ===")
    print(f"config: {config}")
    print(f"batch_size: {batch_size}")
    print(f"frames: {frames}")
    print(f"frequency bins: {freq_bins}")
    print(f"Total FLOPs: {_format(flops)} ({flops / 1e6:.3f} MFLOPs)")
    print(f"Total MACs: {_format(macs)} ({macs / 1e6:.3f} MMAC)")
    print(f"Total params: {_format(params)} ({params / 1e6:.3f} M)")

    if batch_size:
        print(f"Per-sample MACs: {macs / batch_size / 1e6:.3f} MMAC")
    if frames:
        print(f"Per-frame MACs: {macs / max(frames, 1) / 1e6:.3f} MMAC")

    if print_detailed:
        print("\n(calflops detailed breakdown was printed above.)")

        type_param_totals: Dict[str, int] = defaultdict(int)
        type_module_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        accounted_params = 0

        for module_name, module in core_model.named_modules():
            local_params = list(module.named_parameters(recurse=False))
            if not local_params:
                continue
            family = _operation_family(module_name, module)
            for _, param in local_params:
                if not param.requires_grad:
                    continue
                numel = param.numel()
                type_param_totals[family] += numel
                type_module_totals[family][module_name or "<root>"] += numel
                accounted_params += numel

        calflops_params = int(round(float(params))) if params else 0
        table_denominator = accounted_params if accounted_params > 0 else max(calflops_params, 1)

        if calflops_params > accounted_params:
            type_param_totals["Unaccounted"] += calflops_params - accounted_params
            table_denominator = calflops_params

        print("\nParameters by operation type:")
        header = f"{'type':20s} {'params':>15s} {'share (%)':>12s}"
        print(header)
        print("-" * len(header))
        for op_type, count in sorted(type_param_totals.items(), key=lambda item: item[1], reverse=True):
            share = (count / table_denominator * 100.0) if table_denominator > 0 else 0.0
            print(f"{op_type:20s} {_format(float(count)):>15s} {share:12.2f}")
            contribs = type_module_totals.get(op_type)
            if contribs:
                ranked = sorted(contribs.items(), key=lambda item: item[1], reverse=True)[:5]
                details = ", ".join(
                    f"{name or '<root>'}={_format(float(vals))}"
                    for name, vals in ranked
                )
                print(f"    top modules: {details}")

        if calflops_params and calflops_params != accounted_params:
            diff = accounted_params - calflops_params
            direction = "more" if diff > 0 else "fewer"
            print(
                f"Note: module traversal found {abs(diff)} {direction} parameters than calflops reported "
                f"({accounted_params} vs {calflops_params})."
            )

    if latency:
        effective_batch = max(batch_size, 1)
        effective_frames = max(frames, 1)
        input_shape = (effective_batch, 3, effective_frames, freq_bins)
        _profile_latency(
            core_model,
            input_shape,
            device,
            warmup=latency_warmup,
            runs=latency_runs,
            top_k=latency_top_k,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the GTCRN core network with calflops")
    parser.add_argument("--config", type=Path, default=Path("configs/cfg_train.yaml"), help="Config YAML")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy features")
    parser.add_argument("--frames", type=int, default=1, help="Number of STFT frames to simulate")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available (deprecated)")
    parser.add_argument(
        "--print-detailed",
        action="store_true",
        help="Let calflops print its per-layer breakdown",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run profiling on (default: auto)",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Measure latency for the GTCRN core forward pass",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=5,
        metavar="N",
        help="Warmup runs before collecting latency stats (default: 5)",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=20,
        metavar="N",
        help="Measured runs used for latency stats (default: 20)",
    )
    parser.add_argument(
        "--latency-top-k",
        type=int,
        default=20,
        metavar="K",
        help="Maximum number of detailed latency rows to print (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_model(
        config=args.config,
        batch_size=args.batch_size,
        frames=args.frames,
        use_cuda=args.cuda,
        print_detailed=args.print_detailed,
        latency=args.latency,
        latency_warmup=args.latency_warmup,
        latency_runs=args.latency_runs,
        latency_top_k=args.latency_top_k,
        device_pref=args.device,
    )
