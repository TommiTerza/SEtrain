import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.ao.quantization as quant
from omegaconf import OmegaConf
from tqdm import tqdm

from models.gtcrn_end2end import ConvBlock, DPGRNN, GTCRN as Model, GTConvBlock


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


class SampleBuffer:
    def __init__(self, max_values: int, seed: int):
        self.max_values = max(1, int(max_values))
        self.rng = np.random.default_rng(seed)
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


class ActivationCollector:
    def __init__(
        self,
        model: torch.nn.Module,
        sample_per_forward: int,
        max_values_per_module: int,
        seed: int,
    ):
        self.sample_per_forward = max(1, int(sample_per_forward))
        self.max_values_per_module = max(1, int(max_values_per_module))
        self.seed = int(seed)
        self.handles: list[Any] = []
        self.module_to_type: dict[str, str] = {}
        self.buffers: dict[str, SampleBuffer] = {}
        self.block_order: dict[str, list[str]] = {
            "conv": [],
            "gtconv": [],
            "gdrnn": [],
            "deconv": [],
            "gtdeconv": [],
        }

        for module_name, module in model.named_modules():
            block_type = self._get_block_type(module)
            if block_type is None:
                continue
            self.module_to_type[module_name] = block_type
            self.buffers[module_name] = SampleBuffer(
                max_values=self.max_values_per_module,
                seed=self.seed + len(self.buffers),
            )
            self.block_order[block_type].append(module_name)
            self.handles.append(module.register_forward_hook(self._build_hook(module_name)))

    @staticmethod
    def _get_block_type(module: torch.nn.Module) -> str | None:
        if isinstance(module, ConvBlock):
            return "deconv" if module.use_deconv else "conv"
        if isinstance(module, GTConvBlock):
            return "gtdeconv" if module.use_deconv else "gtconv"
        if isinstance(module, DPGRNN):
            return "gdrnn"
        return None

    def _build_hook(self, module_name: str):
        def hook(_, __, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not torch.is_tensor(output):
                return
            flat = output.detach().reshape(-1)
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
            sampled = flat.float().cpu().numpy()
            self.buffers[module_name].add(sampled)

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _resolve_device(device_arg: str) -> torch.device:
    arg = str(device_arg).strip().lower()
    if arg in {"cpu", "none", "-1"}:
        return torch.device("cpu")
    if not arg.isdigit():
        raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'.")
    return torch.device(f"cuda:{int(arg)}" if torch.cuda.is_available() else "cpu")


def _load_float_model(cfg_infer, device: torch.device, keep_fakequant: bool) -> torch.nn.Module:
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
    ignored = sorted(set(raw_network_config.keys()) - set(network_config.keys()))
    if ignored:
        print(
            "Ignoring unsupported network_config keys for current GTCRN implementation: "
            f"{ignored}"
        )

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
    return model


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


def _plot_block_type(
    block_type: str,
    module_names: list[str],
    all_values: dict[str, np.ndarray],
    out_dir: Path,
    bins: int,
) -> None:
    plt = _get_matplotlib_pyplot()
    if not module_names:
        return

    n_modules = len(module_names)
    ncols = min(2, n_modules)
    nrows = (n_modules + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for idx, module_name in enumerate(module_names):
        ax = axes_flat[idx]
        values = all_values[module_name]
        if values.size == 0:
            ax.set_title(f"{module_name}\n(no data)")
            ax.axis("off")
            continue
        ax.hist(values, bins=bins, density=True, alpha=0.85, color="#1f77b4")
        stats = _compute_stats(values)
        ax.axvline(stats["mean"], color="#d62728", linestyle="--", linewidth=1.4, label="mean")
        ax.axvline(stats["p50"], color="#2ca02c", linestyle="-.", linewidth=1.4, label="median")
        ax.set_title(module_name)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    for idx in range(n_modules, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"Activation Distribution: {block_type}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{block_type}_activations.png", dpi=160)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    cfg_infer = OmegaConf.load(args.config)
    noisy_dir = Path(str(cfg_infer.test_dataset.noisy_dir)).expanduser()
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")

    device = _resolve_device(args.device)
    model = _load_float_model(cfg_infer, device=device, keep_fakequant=args.keep_fakequant)

    default_out_dir = Path(str(cfg_infer.network.enh_folder)).expanduser() / (
        "activation_plots_float_" + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
    )
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collector = ActivationCollector(
        model=model,
        sample_per_forward=args.sample_per_forward,
        max_values_per_module=args.max_values_per_module,
        seed=args.seed,
    )
    print(f"Saving plots to: {out_dir}")
    print(f"Using device: {device}")
    print(f"keep_fakequant={args.keep_fakequant}")
    print("Hooked modules:")
    for block_type in ("conv", "gtconv", "gdrnn", "deconv", "gtdeconv"):
        modules = collector.block_order[block_type]
        if modules:
            print(f"  {block_type}:")
            for name in modules:
                print(f"    - {name}")

    noisy_wavs = sorted([p for p in noisy_dir.iterdir() if p.suffix.lower() == ".wav"])
    if not noisy_wavs:
        raise RuntimeError(f"No wav files found in: {noisy_dir}")
    if args.max_files > 0:
        noisy_wavs = noisy_wavs[: args.max_files]

    with torch.inference_mode():
        for wav_path in tqdm(noisy_wavs, desc="Running inference"):
            noisy, _ = sf.read(str(wav_path), dtype="float32")
            if noisy.ndim > 1:
                noisy = noisy[:, 0]
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            _ = model(x)

    collector.remove()

    all_values = {name: buf.finalize() for name, buf in collector.buffers.items()}
    summary = {
        "config": str(Path(args.config).expanduser()),
        "checkpoint": str(cfg_infer.network.checkpoint),
        "noisy_dir": str(noisy_dir),
        "num_files_processed": len(noisy_wavs),
        "device": str(device),
        "keep_fakequant": bool(args.keep_fakequant),
        "sample_per_forward": int(args.sample_per_forward),
        "max_values_per_module": int(args.max_values_per_module),
        "blocks": {},
    }

    for block_type in ("conv", "gtconv", "gdrnn", "deconv", "gtdeconv"):
        module_names = collector.block_order[block_type]
        _plot_block_type(
            block_type=block_type,
            module_names=module_names,
            all_values=all_values,
            out_dir=out_dir,
            bins=args.bins,
        )
        summary["blocks"][block_type] = {}
        for module_name in module_names:
            summary["blocks"][block_type][module_name] = _compute_stats(all_values[module_name])

    with open(out_dir / "activation_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    npz_payload = {name: values for name, values in all_values.items()}
    np.savez_compressed(out_dir / "activation_samples.npz", **npz_payload)

    print(f"Processed {len(noisy_wavs)} files.")
    print(f"Saved stats: {out_dir / 'activation_stats.json'}")
    print(f"Saved samples: {out_dir / 'activation_samples.npz'}")
    print("Saved plots:")
    for block_type in ("conv", "gtconv", "gdrnn", "deconv", "gtdeconv"):
        plot_path = out_dir / f"{block_type}_activations.png"
        if plot_path.exists():
            print(f"  - {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot float-model activation distributions by GTCRN block type."
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
        "--output_dir",
        default="",
        help="Output directory. Default: <enh_folder>/activation_plots_float_<timestamp>",
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
        default=20000,
        help="Max activation values sampled per hooked module on each forward pass.",
    )
    parser.add_argument(
        "--max_values_per_module",
        type=int,
        default=400000,
        help="Upper bound of stored activation values per hooked module.",
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
        help="Random seed for activation sub-sampling.",
    )
    parser.add_argument(
        "--keep_fakequant",
        action="store_true",
        help="If set, keeps fake-quant modules enabled for QAT checkpoints.",
    )

    main(parser.parse_args())
