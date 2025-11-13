#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.gtcrn_end2end import GTCRN


def _try_thop_profile(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    force_disable: bool = False,
):
    if force_disable:
        return None

    try:
        from thop import profile as thop_profile  # type: ignore
    except ImportError:
        return None

    macs, params = thop_profile(model, inputs=inputs, verbose=False)

    per_layer: List[Tuple[str, float, int]] = []
    for name, module in model.named_modules():
        if any(module.children()):
            continue
        total_ops = getattr(module, "total_ops", None)
        if total_ops is None:
            continue
        if isinstance(total_ops, torch.Tensor):
            total_ops = total_ops.item()
        per_layer.append((name or module.__class__.__name__, float(total_ops), sum(p.numel() for p in module.parameters())))

    return macs, params, per_layer


# ---------------------------------------------------------------------------
# Lightweight fallback profiler (covers the modules used in GTCRN)
# ---------------------------------------------------------------------------

HookFn = Callable[[nn.Module, Tuple[Any, ...], Any], None]


def _is_leaf(module: nn.Module) -> bool:
    return not any(module.children())


def _count_conv(module: nn.Conv2d, inputs: Tuple[Any, ...], output: Any) -> float:
    x = inputs[0]
    out = output
    if isinstance(out, tuple):
        out = out[0]
    batch = out.shape[0]
    out_channels = out.shape[1]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
    output_size = out.shape[2] * out.shape[3]
    bias_ops = 0 if module.bias is None else 1
    macs = batch * out_channels * output_size * kernel_ops
    if bias_ops:
        macs += batch * out_channels * output_size
    return macs


def _count_conv_transpose(module: nn.ConvTranspose2d, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    batch = out.shape[0]
    out_channels = out.shape[1]
    kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
    output_size = out.shape[2] * out.shape[3]
    bias_ops = 0 if module.bias is None else 1
    macs = batch * out_channels * output_size * kernel_ops
    if bias_ops:
        macs += batch * out_channels * output_size
    return macs


def _count_linear(module: nn.Linear, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    if out is None:
        out = inputs[0]
    out_features = module.out_features
    batch = out.numel() // out_features
    bias_ops = 0 if module.bias is None else 1
    macs = batch * out_features * module.in_features
    if bias_ops:
        macs += batch * out_features
    return macs


def _count_batchnorm(module: nn.BatchNorm2d, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    numel = out.numel()
    # scale + shift
    return 2.0 * numel


def _count_layernorm(module: nn.LayerNorm, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    numel = out.numel()
    return 5.0 * numel  # mean, variance, scale, shift, epsilon add


def _count_prelu(module: nn.PReLU, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    return float(out.numel())


def _count_sigmoid(module: nn.Sigmoid, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    return 4.0 * out.numel()


def _count_tanh(module: nn.Tanh, inputs: Tuple[Any, ...], output: Any) -> float:
    out = output
    if isinstance(out, tuple):
        out = out[0]
    return 4.0 * out.numel()


def _count_gru(module: nn.GRU, inputs: Tuple[Any, ...], output: Any) -> float:
    x = inputs[0]
    if module.batch_first:
        batch_size, seq_len, input_size = x.shape
    else:
        seq_len, batch_size, input_size = x.shape
    hidden_size = module.hidden_size
    num_layers = module.num_layers
    bidirectional = module.bidirectional
    directions = 2 if bidirectional else 1

    total_macs = 0.0
    current_input = input_size
    for layer in range(num_layers):
        for _ in range(directions):
            mac_per_timestep = 3 * (current_input * hidden_size + hidden_size * hidden_size)
            total_macs += mac_per_timestep * seq_len * batch_size
        current_input = hidden_size * directions
    if module.bias:
        total_macs += seq_len * batch_size * hidden_size * directions * 3
    return total_macs


_COUNTERS: Dict[type, Callable[[nn.Module, Tuple[Any, ...], Any], float]] = {
    nn.Conv2d: _count_conv,
    nn.ConvTranspose2d: _count_conv_transpose,
    nn.Linear: _count_linear,
    nn.BatchNorm2d: _count_batchnorm,
    nn.LayerNorm: _count_layernorm,
    nn.PReLU: _count_prelu,
    nn.Sigmoid: _count_sigmoid,
    nn.Tanh: _count_tanh,
    nn.GRU: _count_gru,
}


def _fallback_profile(model: nn.Module, inputs: Tuple[torch.Tensor, ...]):
    handles = []
    info: Dict[str, Dict[str, Any]] = {}

    def make_hook(name: str, module: nn.Module) -> HookFn:
        def hook(mod: nn.Module, module_inputs: Tuple[Any, ...], module_output: Any):
            counter = None
            for cls, fn in _COUNTERS.items():
                if isinstance(mod, cls):
                    counter = fn
                    break
            macs = 0.0
            if counter is not None:
                macs = float(counter(mod, module_inputs, module_output))
            out_tensor = module_output[0] if isinstance(module_output, tuple) else module_output
            info[name] = {
                "module": mod,
                "macs": macs,
                "params": sum(p.numel() for p in mod.parameters()),
                "output_shape": tuple(out_tensor.shape) if isinstance(out_tensor, torch.Tensor) else None,
            }
        return hook

    for name, module in model.named_modules():
        if _is_leaf(module):
            handles.append(module.register_forward_hook(make_hook(name or module.__class__.__name__, module)))

    model.eval()
    with torch.no_grad():
        model(*inputs)

    for h in handles:
        h.remove()

    per_layer: List[Tuple[str, float, int, Optional[Tuple[int, ...]]]] = []
    total_macs = 0.0
    for name, data in info.items():
        macs = data["macs"]
        params = data["params"]
        shape = data["output_shape"]
        if macs == 0 and params == 0:
            continue
        total_macs += macs
        per_layer.append((name, macs, params, shape))

    params_total = sum(p.numel() for p in model.parameters())
    return total_macs, params_total, per_layer


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return f"{value:.3f}"


def profile_model(
    cfg_path: Path,
    batch_size: Optional[int],
    seconds: Optional[float],
    use_cuda: bool,
    force_fallback: bool,
) -> None:
    cfg = OmegaConf.load(str(cfg_path))
    model_kwargs = dict(cfg.network_config)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    model = GTCRN(**model_kwargs).to(device)

    sample_rate = cfg.get("samplerate", 16000)
    length_sec = seconds if seconds is not None else cfg.test_dataset.length_in_seconds
    bs = batch_size if batch_size is not None else cfg.test_dataloader.batch_size
    num_samples = int(sample_rate * length_sec)

    dummy = torch.randn(bs, num_samples, device=device)

    thop_result = _try_thop_profile(model, (dummy,), force_disable=force_fallback)
    if thop_result is not None:
        total_macs, total_params, per_layer_raw = thop_result
        per_layer = [(name, macs, params, None) for name, macs, params in per_layer_raw]
        if not per_layer or all(macs == 0 for _, macs, _ in per_layer_raw):
            total_macs, total_params, per_layer = _fallback_profile(model, (dummy,))
    else:
        total_macs, total_params, per_layer = _fallback_profile(model, (dummy,))

    total_mmac = total_macs / 1e6
    total_params_m = total_params / 1e6

    print("\n=== GTCRN MAC Profile ===")
    print(f"Config: {cfg_path}")
    print(f"Batch size: {bs}, seconds per sample: {length_sec}, samples per clip: {num_samples}")
    print(f"Total MACs: {total_mmac:.3f} MMACs  (per sample: {total_mmac/bs:.3f} MMACs)")
    print(f"Total parameters: {total_params_m:.3f} M params")

    header = f"{'Layer':50s} {'Type':18s} {'Output Shape':20s} {'Params':>12s} {'MACs (MM)':>12s}"
    print("\n" + header)
    print("-" * len(header))

    per_layer_sorted = sorted(per_layer, key=lambda item: item[1], reverse=True)
    for name, macs, params, shape in per_layer_sorted:
        module = dict(model.named_modules()).get(name, None)
        mtype = module.__class__.__name__ if module is not None else "-"
        shape_str = str(shape) if shape is not None else "-"
        print(f"{name:50s} {mtype:18s} {shape_str:20s} {params:12d} {macs / 1e6:12.3f}")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile GTCRN MACs via thop")
    parser.add_argument("--config", default="configs/cfg_train.yaml", type=Path, help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seconds", type=float, default=None, help="Override clip duration in seconds")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip thop and use the built-in operation counter",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_model(args.config, args.batch_size, args.seconds, args.cuda, args.force_fallback)
