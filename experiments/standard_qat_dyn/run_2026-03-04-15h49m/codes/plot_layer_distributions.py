import argparse
import inspect
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.ao.quantization as quant

from models.gtcrn_end2end import GTCRN as Model


def _require(module: str, install_hint: str) -> Any:
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{module} is required. {install_hint}") from exc


def _get_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc
    return plt


def _extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format. Expected dict-like checkpoint.")
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not find model weights in checkpoint ('model' or 'state_dict').")


def _clean_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _resolve_device(device_arg: str) -> torch.device:
    arg = str(device_arg).strip().lower()
    if arg in {"cpu", "none", "-1"}:
        return torch.device("cpu")
    if not arg.isdigit():
        raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'.")
    return torch.device(f"cuda:{int(arg)}" if torch.cuda.is_available() else "cpu")


def _compile_patterns(raw: Iterable[str]) -> list[re.Pattern[str]]:
    patterns = []
    for value in raw:
        value = str(value).strip()
        if not value:
            continue
        patterns.append(re.compile(value))
    return patterns


def _matches_any(patterns: list[re.Pattern[str]], value: str) -> bool:
    return any(p.search(value) is not None for p in patterns)


class SampleBuffer:
    def __init__(self, max_values: int, seed: int):
        self.max_values = max(1, int(max_values))
        self.rng = np.random.default_rng(int(seed))
        self.chunks: list[np.ndarray] = []
        self.num_cached = 0
        self.num_seen = 0

    def add(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        self.chunks.append(values)
        self.num_cached += values.size
        self.num_seen += values.size
        if self.num_cached > self.max_values * 3:
            self._compress()

    def _compress(self) -> None:
        if not self.chunks:
            return
        merged = np.concatenate(self.chunks, axis=0)
        if merged.size > self.max_values:
            idx = self.rng.choice(merged.size, size=self.max_values, replace=False)
            merged = merged[idx]
        self.chunks = [merged]
        self.num_cached = merged.size

    def finalize(self) -> np.ndarray:
        self._compress()
        if not self.chunks:
            return np.zeros((0,), dtype=np.float32)
        return self.chunks[0]


class InputActivationCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        module_names: list[str],
        sample_per_forward: int,
        max_values_per_module: int,
        seed: int,
    ):
        self.sample_per_forward = max(1, int(sample_per_forward))
        self.module_names = list(module_names)
        self.handles: list[Any] = []
        self.buffers: dict[str, SampleBuffer] = {}
        named_modules = dict(model.named_modules())
        missing = [name for name in self.module_names if name not in named_modules]
        if missing:
            raise RuntimeError(
                "Cannot hook requested modules. Missing modules: "
                f"{missing[:10]}"
            )

        for idx, name in enumerate(self.module_names):
            self.buffers[name] = SampleBuffer(max_values=max_values_per_module, seed=seed + idx)
            module = named_modules[name]
            self.handles.append(module.register_forward_hook(self._build_hook(name)))

    def _build_hook(self, module_name: str):
        def hook(_, inputs, __):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            flat = x.detach().reshape(-1)
            if flat.numel() == 0:
                return
            if flat.numel() > self.sample_per_forward:
                idx = torch.randint(
                    low=0,
                    high=flat.numel(),
                    size=(self.sample_per_forward,),
                    device=flat.device,
                )
                flat = flat[idx]
            self.buffers[module_name].add(flat.float().cpu().numpy())

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _compute_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
    }


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return safe.strip("._") or "layer"


def _gather_weight_values(
    module: torch.nn.Module,
    *,
    max_values: int,
    seed: int,
) -> np.ndarray:
    weights: list[torch.Tensor] = []
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        weights.append(module.weight)
    elif isinstance(module, torch.nn.GRU):
        for name, param in module.named_parameters(recurse=False):
            if name.startswith("weight_"):
                weights.append(param)
    else:
        return np.zeros((0,), dtype=np.float32)

    if not weights:
        return np.zeros((0,), dtype=np.float32)

    flat = torch.cat([w.detach().reshape(-1).cpu() for w in weights], dim=0).numpy().astype(np.float32, copy=False)
    if (max_values > 0) and (flat.size > max_values):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(flat.size, size=max_values, replace=False)
        flat = flat[idx]
    return flat


def _select_layer_names(
    model: torch.nn.Module,
    scope: str,
    include: list[re.Pattern[str]],
    exclude: list[re.Pattern[str]],
) -> list[str]:
    names: list[str] = []
    want_conv = scope in {"all", "conv"}
    want_gru = scope in {"all", "gru"}
    for name, module in model.named_modules():
        if not name:
            continue
        is_conv = isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
        is_gru = isinstance(module, torch.nn.GRU)
        if (is_conv and not want_conv) or (is_gru and not want_gru):
            continue
        if not is_conv and not is_gru:
            continue
        if include and not _matches_any(include, name):
            continue
        if exclude and _matches_any(exclude, name):
            continue
        names.append(name)
    return names


def _load_model_from_infer_cfg(cfg_path: str, device: torch.device, keep_fakequant: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    OmegaConf = _require("omegaconf", "Install it with: pip install omegaconf").OmegaConf

    cfg_infer = OmegaConf.load(cfg_path)
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    qat_cfg = cfg_network["qat"] if "qat" in cfg_network else {}
    qat_enabled = bool(qat_cfg.get("enabled", False))
    scale_constraint_cfg = qat_cfg.get("scale_constraint", {})
    if isinstance(scale_constraint_cfg, dict):
        scale_constraint_mode = scale_constraint_cfg.get("mode", qat_cfg.get("scale_constraint_mode", "none"))
        scale_constraint_frac_bits = scale_constraint_cfg.get(
            "frac_bits", qat_cfg.get("scale_constraint_frac_bits", None)
        )
        scale_constraint_pow2_rounding = scale_constraint_cfg.get(
            "pow2_rounding", qat_cfg.get("scale_constraint_pow2_rounding", "nearest")
        )
    else:
        scale_constraint_mode = scale_constraint_cfg if scale_constraint_cfg is not None else qat_cfg.get("scale_constraint_mode", "none")
        scale_constraint_frac_bits = qat_cfg.get("scale_constraint_frac_bits", None)
        scale_constraint_pow2_rounding = qat_cfg.get("scale_constraint_pow2_rounding", "nearest")

    raw_network_config = dict(cfg_network["network_config"])
    accepted = set(inspect.signature(Model.__init__).parameters.keys()) - {"self"}
    network_config = {k: v for k, v in raw_network_config.items() if k in accepted}

    model = Model(**network_config).to(device)
    if qat_enabled:
        model.prepare_qat(
            backend=qat_cfg.get("backend", "fbgemm"),
            quantize_deconv=bool(qat_cfg.get("quantize_deconv", False)),
            per_channel_weights=bool(qat_cfg.get("per_channel_weights", False)),
            quantize_gru=bool(qat_cfg.get("qat_gru", False)),
            scale_constraint_mode=scale_constraint_mode,
            scale_constraint_frac_bits=scale_constraint_frac_bits,
            scale_constraint_pow2_rounding=scale_constraint_pow2_rounding,
        )

    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    state_dict = _clean_state_dict_keys(_extract_model_state_dict(checkpoint))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"Checkpoint incompatible. Missing keys: {missing_keys[:10]}")

    ignored_prefixes = ("activation_post_process", "weight_fake_quant")
    bad_unexpected = [k for k in unexpected_keys if not any(pfx in k for pfx in ignored_prefixes)]
    if bad_unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {bad_unexpected[:10]}")

    model.eval()
    if qat_enabled and not keep_fakequant:
        model.apply(quant.disable_observer)
        model.apply(quant.disable_fake_quant)

    meta = {
        "cfg_infer": cfg_infer,
        "cfg_network_path": str(cfg_infer.network.config),
        "checkpoint": str(cfg_infer.network.checkpoint),
        "qat_enabled": qat_enabled,
        "keep_fakequant": bool(keep_fakequant),
    }
    return model, meta


def _plot_one(
    module_name: str,
    act_values: np.ndarray,
    weight_values: np.ndarray,
    out_dir: Path,
    bins: int,
) -> Path:
    plt = _get_matplotlib_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    def plot_hist(ax, values: np.ndarray, title: str, xlabel: str):
        if values.size == 0:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            return
        ax.hist(values, bins=bins, density=True, alpha=0.85, color="#1f77b4")
        stats = _compute_stats(values)
        ax.axvline(stats["mean"], color="#d62728", linestyle="--", linewidth=1.4, label="mean")
        ax.axvline(stats["p50"], color="#2ca02c", linestyle="-.", linewidth=1.4, label="median")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    plot_hist(axes[0], act_values, "Input Activations", "Activation")
    plot_hist(axes[1], weight_values, "Weights", "Weight")

    fig.suptitle(module_name, fontsize=11)
    fig.tight_layout()
    out_path = out_dir / f"{_sanitize_filename(module_name)}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(args: argparse.Namespace) -> None:
    tqdm = _require("tqdm", "Install it with: pip install tqdm").tqdm
    sf = _require("soundfile", "Install it with: pip install soundfile")
    OmegaConf = _require("omegaconf", "Install it with: pip install omegaconf").OmegaConf

    cfg_infer = OmegaConf.load(args.config)
    noisy_dir = Path(str(cfg_infer.test_dataset.noisy_dir)).expanduser()
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")

    device = _resolve_device(args.device)
    model, meta = _load_model_from_infer_cfg(args.config, device=device, keep_fakequant=args.keep_fakequant)

    include = _compile_patterns(args.include)
    exclude = _compile_patterns(args.exclude)
    module_names = _select_layer_names(model, args.scope, include=include, exclude=exclude)
    if not module_names:
        raise RuntimeError("No modules matched selection. Try a different --scope/--include/--exclude.")

    default_out_dir = Path(str(cfg_infer.network.enh_folder)).expanduser() / (
        "layer_distributions_" + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
    )
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collector = InputActivationCollector(
        model=model,
        module_names=module_names,
        sample_per_forward=args.sample_per_forward,
        max_values_per_module=args.max_values_per_module,
        seed=args.seed,
    )

    noisy_wavs = sorted([p for p in noisy_dir.iterdir() if p.suffix.lower() == ".wav"])
    if not noisy_wavs:
        raise RuntimeError(f"No wav files found in: {noisy_dir}")
    if args.max_files > 0:
        noisy_wavs = noisy_wavs[: args.max_files]

    print(f"Saving plots to: {out_dir}")
    print(f"Using device: {device}")
    print(f"scope={args.scope} layers={len(module_names)} keep_fakequant={meta['keep_fakequant']}")

    with torch.inference_mode():
        for wav_path in tqdm(noisy_wavs, desc="Running inference"):
            noisy, _ = sf.read(str(wav_path), dtype="float32")
            if getattr(noisy, "ndim", 1) > 1:
                noisy = noisy[:, 0]
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            _ = model(x)

    collector.remove()

    act_samples = {name: buf.finalize() for name, buf in collector.buffers.items()}
    weight_samples = {}
    for idx, name in enumerate(module_names):
        module = dict(model.named_modules())[name]
        weight_samples[name] = _gather_weight_values(
            module,
            max_values=args.max_weight_values,
            seed=args.seed + 10_000 + idx,
        )

    stats = {
        "config": str(Path(args.config).expanduser()),
        "checkpoint": str(meta["checkpoint"]),
        "noisy_dir": str(noisy_dir),
        "num_files_processed": len(noisy_wavs),
        "device": str(device),
        "scope": str(args.scope),
        "keep_fakequant": bool(meta["keep_fakequant"]),
        "sample_per_forward": int(args.sample_per_forward),
        "max_values_per_module": int(args.max_values_per_module),
        "max_weight_values": int(args.max_weight_values),
        "bins": int(args.bins),
        "modules": {},
    }

    plot_paths = []
    for name in module_names:
        stats["modules"][name] = {
            "activation": _compute_stats(act_samples[name]),
            "weights": _compute_stats(weight_samples[name]),
        }
        plot_paths.append(_plot_one(name, act_samples[name], weight_samples[name], out_dir=out_dir, bins=args.bins))

    with open(out_dir / "layer_distribution_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    np.savez_compressed(out_dir / "activation_input_samples.npz", **act_samples)
    np.savez_compressed(out_dir / "weight_samples.npz", **weight_samples)

    print(f"Processed {len(noisy_wavs)} files.")
    print(f"Saved stats: {out_dir / 'layer_distribution_stats.json'}")
    print(f"Saved activation samples: {out_dir / 'activation_input_samples.npz'}")
    print(f"Saved weight samples: {out_dir / 'weight_samples.npz'}")
    print("Saved plots:")
    for path in plot_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot input activation and weight distributions for selected GTCRN layers."
    )
    parser.add_argument(
        "-C",
        "--config",
        default="configs/cfg_infer.yaml",
        help="Inference config file (same format used by infer.py).",
    )
    parser.add_argument(
        "-D",
        "--device",
        default="0",
        help="GPU index (e.g. 0) or 'cpu'.",
    )
    parser.add_argument(
        "--scope",
        default="all",
        choices=("all", "conv", "gru"),
        help="Which layers to include: all=conv+gru, conv=Conv2d+ConvTranspose2d, gru=all nn.GRU (TRA included).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Regex filter for module names (repeatable). If set, only matching modules are kept.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Regex exclude for module names (repeatable).",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory. Default: <enh_folder>/layer_distributions_<timestamp>",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum number of noisy wav files to process (0 means all).",
    )
    parser.add_argument(
        "--sample_per_forward",
        type=int,
        default=4000,
        help="Max input activation values sampled per hooked module on each forward pass.",
    )
    parser.add_argument(
        "--max_values_per_module",
        type=int,
        default=200000,
        help="Upper bound of stored input activation values per module.",
    )
    parser.add_argument(
        "--max_weight_values",
        type=int,
        default=200000,
        help="Upper bound of stored weight values per module (0 keeps all).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Histogram bins for each subplot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Random seed for sub-sampling.",
    )
    parser.add_argument(
        "--keep_fakequant",
        action="store_true",
        help="If set, keeps fake-quant modules enabled for QAT checkpoints.",
    )

    main(parser.parse_args())
