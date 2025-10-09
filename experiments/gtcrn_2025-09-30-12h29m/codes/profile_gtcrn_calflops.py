#!/usr/bin/env python3
"""Profile the core GTCRN network (without FFT) using calflops."""
import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
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


def _iter_leaf_modules(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Module]]:
    """Yield all named leaf modules of ``model``."""

    for name, module in model.named_modules():
        if not name:
            continue
        if any(module.children()):
            continue
        yield name, module


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

    leaves = list(_iter_leaf_modules(model))
    if not leaves:
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

    with torch.inference_mode():
        for _ in range(max(warmup, 0)):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    handles = []
    for name, module in leaves:
        handles.append(module.register_forward_pre_hook(make_pre_hook(name)))
        handles.append(module.register_forward_hook(make_post_hook(name)))

    totals: List[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_start = time.perf_counter()
            model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            totals.append(time.perf_counter() - total_start)

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

    group_totals: Dict[str, float] = defaultdict(float)
    group_calls: Dict[str, float] = defaultdict(float)

    module_stats = []
    for name, durations in latencies.items():
        if not durations:
            continue
        total_time = sum(durations)
        calls = len(durations)
        per_run = total_time / runs
        per_call = total_time / calls
        std = statistics.pstdev(durations) if calls > 1 else 0.0
        share = (per_run / total_mean * 100.0) if total_mean > 0 else 0.0
        module_stats.append((name, per_run, per_call, std, calls, share))

        top_level = name.split(".", 1)[0]
        group_totals[top_level] += total_time
        group_calls[top_level] += calls

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
