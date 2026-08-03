"""
PTQ Qm.n sweep utility for GTCRN.

What this script does:
1) Loads a float GTCRN checkpoint.
2) Builds a calibration-only PTQ flow (observer collection, no gradient updates).
3) Sweeps Qm.n formats for selected bit-widths (e.g., int8 and int16 emulation).
4) Reports intrusive metrics (PESQ/ESTOI/SISNR/SDR) for each format.

Notes:
- int8 can optionally be evaluated as a real converted quantized model.
- int16 is evaluated in fake-quant emulation mode because eager PyTorch
  quantized kernels are int8-oriented.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.ao.quantization as quant
from omegaconf import OmegaConf
from pesq import PesqError, pesq
from pystoi import stoi
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from tqdm import tqdm

from models.gtcrn_end2end import GTCRN as Model


SUPPORTED_METRICS = ("pesq", "estoi", "sisnr", "sdr")


def normalize_cfg_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def load_sweep_arg_defaults(
    cfg_path: str | None,
    *,
    valid_keys: set[str] | None = None,
) -> dict[str, object]:
    if cfg_path is None:
        return {}

    path = Path(cfg_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Sweep config not found: {path}")

    raw_cfg = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if raw_cfg is None:
        return {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Sweep config must resolve to a mapping, got: {type(raw_cfg).__name__}")

    if "ptq_qmn_sweep" in raw_cfg:
        raw_cfg = raw_cfg["ptq_qmn_sweep"]
        if not isinstance(raw_cfg, dict):
            raise ValueError("Sweep config key 'ptq_qmn_sweep' must be a mapping.")

    defaults = {normalize_cfg_key(k): v for k, v in raw_cfg.items()}
    if valid_keys is not None:
        unknown = sorted(k for k in defaults.keys() if k not in valid_keys)
        if unknown:
            raise ValueError(
                f"Unsupported keys in sweep config {path}: {unknown}. "
                "Use argparse destination names (snake_case) or option names (kebab-case)."
            )
    return defaults


def parse_csv_ints(text: str) -> list[int]:
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def normalize_mode_flag(value: str | None) -> str:
    if value is None:
        return "none"
    return str(value).strip().lower()


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format in: {checkpoint_path}")

    if state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def metric_sisnr(ref: np.ndarray, inf: np.ndarray) -> float:
    inf = inf - inf.mean()
    ref = ref - ref.mean()
    a = np.sum(inf * ref) / np.sum(ref**2 + 1e-8)
    e_tgt = a * ref
    e_res = inf - e_tgt
    return float(10.0 * np.log10((np.sum(e_tgt**2) + 1e-8) / (np.sum(e_res**2) + 1e-8)))


def metric_sdr(ref: np.ndarray, inf: np.ndarray) -> float:
    inf = inf - inf.mean()
    ref = ref - ref.mean()
    e_tgt = ref
    e_res = inf - e_tgt
    return float(10.0 * np.log10((np.sum(e_tgt**2) + 1e-8) / (np.sum(e_res**2) + 1e-8)))


def metric_pesq(ref: np.ndarray, inf: np.ndarray, sample_rate: int) -> float:
    if sample_rate == 8000:
        mode = "nb"
    elif sample_rate == 16000:
        mode = "wb"
    else:
        return float("nan")

    try:
        score = pesq(sample_rate, ref, inf, mode=mode, on_error=PesqError.RETURN_VALUES)
    except Exception:
        return float("nan")
    if score == PesqError.NO_UTTERANCES_DETECTED:
        return float("nan")
    return float(score)


def metric_estoi(ref: np.ndarray, inf: np.ndarray, sample_rate: int) -> float:
    try:
        return float(stoi(ref, inf, fs_sig=sample_rate, extended=True))
    except Exception:
        return float("nan")


def collect_wav_pairs(noisy_dir: Path, clean_dir: Path) -> list[tuple[Path, Path]]:
    noisy_wavs = sorted([p for p in noisy_dir.iterdir() if p.suffix.lower() == ".wav"])
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for noisy_path in noisy_wavs:
        clean_path = clean_dir / noisy_path.name
        if clean_path.exists():
            pairs.append((noisy_path, clean_path))
        else:
            missing += 1
    if missing > 0:
        print(f"Warning: skipped {missing} noisy files without matching clean reference.")
    return pairs


def set_fake_quant_bit_width(model: torch.nn.Module, bit_width: int) -> int:
    if bit_width < 2:
        raise ValueError(f"bit_width must be >= 2, got {bit_width}")

    signed_qmin = -(1 << (bit_width - 1))
    signed_qmax = (1 << (bit_width - 1)) - 1
    unsigned_qmin = 0
    unsigned_qmax = (1 << bit_width) - 1

    updated = 0
    for module in model.modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        observer = getattr(module, "activation_post_process", None)

        is_signed = int(getattr(module, "quant_min", 0)) < 0
        if observer is not None:
            obs_dtype = getattr(observer, "dtype", None)
            if obs_dtype in {torch.qint8, torch.int8, torch.qint32}:
                is_signed = True

        qmin, qmax = (signed_qmin, signed_qmax) if is_signed else (unsigned_qmin, unsigned_qmax)
        module.quant_min = int(qmin)
        module.quant_max = int(qmax)

        if observer is not None:
            if hasattr(observer, "quant_min"):
                observer.quant_min = int(qmin)
            if hasattr(observer, "quant_max"):
                observer.quant_max = int(qmax)
        updated += 1
    return updated


def _strip_fake_quant_suffix(module_name: str) -> str:
    for suffix in (".weight_fake_quant", ".activation_post_process"):
        if module_name.endswith(suffix):
            return module_name[: -len(suffix)]
    return module_name


def _is_group_in_scope(group_name: str, scope: str) -> bool:
    scope = str(scope).strip().lower()
    if scope == "all":
        return True

    lname = group_name.lower()
    if scope == "conv":
        return "conv" in lname
    if scope == "gru":
        return ("gru" in lname) or ("rnn" in lname)
    raise ValueError(f"Unsupported per-layer scope: {scope}")


def collect_fake_quant_groups(
    model: torch.nn.Module,
    scope: str = "all",
) -> dict[str, list[tuple[str, FakeQuantizeBase]]]:
    groups: dict[str, list[tuple[str, FakeQuantizeBase]]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        group_name = _strip_fake_quant_suffix(module_name)
        if not _is_group_in_scope(group_name, scope):
            continue
        groups.setdefault(group_name, []).append((module_name, module))
    return {name: groups[name] for name in sorted(groups)}


def apply_group_frac_bits(
    model: torch.nn.Module,
    group_frac_bits: dict[str, int],
) -> int:
    """
    Apply per-group frac-bits directly on constrained observers.
    """
    updated = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        group_name = _strip_fake_quant_suffix(module_name)
        if group_name not in group_frac_bits:
            continue
        observer = getattr(module, "activation_post_process", None)
        if observer is None:
            continue
        if hasattr(observer, "scale_constraint_mode"):
            observer.scale_constraint_mode = "fixed"
        if hasattr(observer, "scale_constraint_frac_bits"):
            observer.scale_constraint_frac_bits = int(group_frac_bits[group_name])
        if hasattr(observer, "scale_constraint_pow2_rounding"):
            observer.scale_constraint_pow2_rounding = "nearest"
        updated += 1
    return updated


def evaluate_model(
    model: torch.nn.Module,
    eval_pairs: list[tuple[Path, Path]],
    device: torch.device,
    metrics: tuple[str, ...],
    expected_sample_rate: int | None,
) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {m: [] for m in metrics}
    total_audio_sec = 0.0
    start_time = time.time()

    model.eval()
    with torch.inference_mode():
        for noisy_path, clean_path in tqdm(eval_pairs, desc="eval", leave=False):
            noisy, sr_noisy = load_wav_mono(noisy_path)
            clean, sr_clean = load_wav_mono(clean_path)
            if sr_noisy != sr_clean:
                raise RuntimeError(f"Sample rate mismatch: {noisy_path} ({sr_noisy}) vs {clean_path} ({sr_clean})")
            if (expected_sample_rate is not None) and (sr_noisy != expected_sample_rate):
                raise RuntimeError(
                    f"Unexpected sample rate {sr_noisy} for {noisy_path}. "
                    f"Expected {expected_sample_rate}."
                )

            inp = torch.from_numpy(noisy).unsqueeze(0).to(device)
            enhanced = model(inp).detach().cpu().numpy().reshape(-1).astype(np.float32)

            n = min(clean.shape[0], enhanced.shape[0])
            if n <= 0:
                continue
            clean = clean[:n]
            enhanced = enhanced[:n]
            total_audio_sec += float(n) / float(sr_noisy)

            if "pesq" in metrics:
                metric_values["pesq"].append(metric_pesq(clean, enhanced, sr_noisy))
            if "estoi" in metrics:
                metric_values["estoi"].append(metric_estoi(clean, enhanced, sr_noisy))
            if "sisnr" in metrics:
                metric_values["sisnr"].append(metric_sisnr(clean, enhanced))
            if "sdr" in metrics:
                metric_values["sdr"].append(metric_sdr(clean, enhanced))

    elapsed = time.time() - start_time
    summary: dict[str, float] = {
        "num_eval_files": float(len(eval_pairs)),
        "audio_seconds": total_audio_sec,
        "wall_time_seconds": elapsed,
        "rtf": elapsed / total_audio_sec if total_audio_sec > 0 else float("nan"),
    }
    for metric_name, values in metric_values.items():
        summary[metric_name] = float(np.nanmean(values)) if values else float("nan")
    return summary


def build_frac_bits_by_bit(
    bit_widths: list[int],
    frac_bits_override: list[int] | None,
) -> dict[int, list[int]]:
    frac_by_bit: dict[int, list[int]] = {}
    for bits in bit_widths:
        if bits < 2:
            raise ValueError(f"bit-width must be >= 2, got {bits}")
        if frac_bits_override is None:
            frac_bits_values = list(range(bits))
        else:
            frac_bits_values = sorted(set(int(n) for n in frac_bits_override if 0 <= int(n) < bits))
            if not frac_bits_values:
                raise ValueError(
                    f"Provided frac-bits list {frac_bits_override} has no valid value for bit-width={bits}."
                )
        frac_by_bit[int(bits)] = frac_bits_values
    return frac_by_bit


def build_qmn_formats(bit_widths: list[int], frac_bits_override: list[int] | None) -> list[dict[str, int | str]]:
    formats: list[dict[str, int | str]] = []
    frac_by_bit = build_frac_bits_by_bit(bit_widths, frac_bits_override)
    for bits, frac_bits_values in frac_by_bit.items():
        for n in frac_bits_values:
            m = bits - 1 - n
            formats.append({"bits": bits, "m": m, "n": n, "label": f"Q{m}.{n}"})
    return formats


def split_calib_eval_pairs(
    pairs: list[tuple[Path, Path]],
    calib_count: int,
    eval_count: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if calib_count < 1:
        raise ValueError(f"calib_count must be >= 1, got {calib_count}")
    if eval_count == 0 or eval_count < -1:
        raise ValueError(f"eval_count must be -1 or >= 1, got {eval_count}")
    if len(pairs) <= calib_count:
        raise ValueError(
            f"Need more files than calib_count. files={len(pairs)}, calib_count={calib_count}"
        )

    calib_pairs = pairs[:calib_count]
    remaining = pairs[calib_count:]
    if eval_count == -1:
        eval_pairs = remaining
    else:
        eval_pairs = remaining[: min(eval_count, len(remaining))]
    if not eval_pairs:
        raise ValueError("No evaluation files selected. Increase dataset or reduce calib_count.")
    return calib_pairs, eval_pairs


def build_prepared_model(
    network_cfg: dict,
    state_dict: dict[str, torch.Tensor],
    *,
    device: torch.device,
    backend: str,
    quantize_conv: bool,
    quantize_deconv: bool,
    per_channel_weights: bool,
    quantize_linear: bool,
    quantize_gru: bool,
    frac_bits: int,
) -> torch.nn.Module:
    model = Model(**network_cfg).to(device)
    strict_loaded = True
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        strict_loaded = False
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "Warning: checkpoint loaded with strict=False "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
    if strict_loaded:
        print("Loaded checkpoint with strict=True.")

    model.prepare_qat(
        backend=backend,
        quantize_conv=quantize_conv,
        quantize_deconv=quantize_deconv,
        per_channel_weights=per_channel_weights,
        quantize_linear=quantize_linear,
        quantize_gru=quantize_gru,
        scale_constraint_mode="fixed",
        scale_constraint_frac_bits=int(frac_bits),
        scale_constraint_pow2_rounding="nearest",
    )
    return model


def build_float_model(
    network_cfg: dict,
    state_dict: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.nn.Module:
    model = Model(**network_cfg).to(device)
    strict_loaded = True
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        strict_loaded = False
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "Warning: checkpoint loaded with strict=False "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
    if strict_loaded:
        print("Loaded checkpoint with strict=True.")
    return model


def sanitize_network_cfg_for_model(network_cfg: dict) -> tuple[dict, list[str]]:
    """
    Keep only GTCRN.__init__ kwargs accepted by this codebase.

    This allows running PTQ on legacy experiment configs that contain
    keys not supported by the current model implementation.
    """
    signature = inspect.signature(Model.__init__)
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }

    sanitized = {}
    dropped = []
    for key, value in network_cfg.items():
        if key in allowed:
            sanitized[key] = value
        else:
            dropped.append(str(key))
    return sanitized, sorted(dropped)


def run_calibration(
    model: torch.nn.Module,
    calib_pairs: list[tuple[Path, Path]],
    *,
    device: torch.device,
    expected_sample_rate: int | None,
):
    model.eval()
    model.apply(quant.disable_fake_quant)
    model.apply(quant.enable_observer)

    with torch.inference_mode():
        for noisy_path, _ in tqdm(calib_pairs, desc="calib", leave=False):
            noisy, sr = load_wav_mono(noisy_path)
            if (expected_sample_rate is not None) and (sr != expected_sample_rate):
                raise RuntimeError(
                    f"Unexpected sample rate {sr} for {noisy_path}. Expected {expected_sample_rate}."
                )
            inp = torch.from_numpy(noisy).unsqueeze(0).to(device)
            _ = model(inp)

    model.apply(quant.disable_observer)
    model.apply(quant.enable_fake_quant)


def _format_tensor_values(values: torch.Tensor, *, max_elems: int = 8) -> str:
    flat = values.detach().cpu().reshape(-1)
    if flat.numel() == 0:
        return "[]"
    if flat.numel() == 1:
        value = flat.item()
        if isinstance(value, float):
            return f"{float(value):.6g}"
        return str(int(value))
    prefix = ", ".join(str(int(v)) if float(v).is_integer() else f"{float(v):.6g}" for v in flat[:max_elems].tolist())
    if flat.numel() > max_elems:
        return f"[{prefix}, ...] (len={flat.numel()})"
    return f"[{prefix}]"


def print_calibration_offsets(
    model: torch.nn.Module,
    *,
    scope: str,
):
    """
    Print calibration-derived offsets (zero_points).

    Notes:
    - Offsets are not global; they are per-quantizer (typically per-layer).
    - For activation affine quantization, zero_point depends on observed min/max.
    - For symmetric quantization (common for weights), zero_point is usually 0.
    """
    scope = str(scope).strip().lower()
    if scope not in {"auto", "gru", "fakequant"}:
        raise ValueError("calib_offsets_scope must be one of: auto|gru|fakequant")

    if scope in {"auto", "gru"}:
        rows = []
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.GRU):
                continue
            qparams = Model._extract_gru_input_qparams(module)
            rows.append(
                (
                    str(module_name),
                    float(qparams.get("scale", 1.0)),
                    int(qparams.get("zero_point", 0)),
                    int(qparams.get("quant_min", 0)),
                    int(qparams.get("quant_max", 255)),
                )
            )
        if rows:
            print("  Calib offsets (GRU input quantizer):")
            for name, scale, zero_point, qmin, qmax in rows:
                print(f"    {name}: zero_point={zero_point}, scale={scale:.6g}, qrange=[{qmin},{qmax}]")
        else:
            print("  Calib offsets (GRU input quantizer): none (no nn.GRU modules found).")

        if scope == "gru":
            return

    print("  Calib offsets (FakeQuantize modules):")
    entries = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        scale = getattr(module, "scale", None)
        zero_point = getattr(module, "zero_point", None)
        if (scale is None) or (zero_point is None) or (scale.numel() == 0) or (zero_point.numel() == 0):
            observer = getattr(module, "activation_post_process", None)
            if observer is not None and hasattr(observer, "calculate_qparams"):
                try:
                    scale, zero_point = observer.calculate_qparams()
                except Exception:
                    scale, zero_point = None, None

        if (scale is None) or (zero_point is None):
            continue

        qmin = int(getattr(module, "quant_min", 0))
        qmax = int(getattr(module, "quant_max", 255))
        kind = "weight" if str(module_name).endswith(".weight_fake_quant") else "activation"
        print(
            f"    {module_name} ({kind}): "
            f"zero_point={_format_tensor_values(zero_point)}, "
            f"scale={_format_tensor_values(scale)}, "
            f"qrange=[{qmin},{qmax}]"
        )
        entries += 1
    if entries == 0:
        print("    (none)")


def save_results(
    rows: list[dict[str, object]],
    out_dir: Path,
    run_meta: dict[str, object],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ptq_qmn_results.csv"
    json_path = out_dir / "ptq_qmn_results.json"

    fields = [
        "sweep_mode",
        "bits",
        "q_format",
        "base_q_format",
        "m",
        "n",
        "base_n",
        "target_group",
        "eval_mode",
        "num_eval_files",
        "pesq",
        "estoi",
        "sisnr",
        "sdr",
        "rtf",
        "wall_time_seconds",
        "audio_seconds",
        "calib_seconds",
        "calib_files",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    payload = {"meta": run_meta, "results": rows}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


def print_ranked_summary(rows: list[dict[str, object]], metrics: tuple[str, ...]):
    if not rows:
        print("No rows to summarize.")
        return

    rank_metric = "pesq" if "pesq" in metrics else metrics[0]
    ranked = sorted(rows, key=lambda x: float(x.get(rank_metric, float("-inf"))), reverse=True)
    print("")
    print(f"Top formats by {rank_metric.upper()}:")
    for row in ranked[: min(10, len(ranked))]:
        print(
            f"  {row['q_format']:>6} ({row['bits']}b, mode={row['eval_mode']}): "
            + ", ".join(
                f"{m}={float(row[m]):.4f}" for m in metrics if m in row and not math.isnan(float(row[m]))
            )
            + f", rtf={float(row.get('rtf', float('nan'))):.4f}"
        )


def main(args):
    if args.num_threads is not None and args.num_threads > 0:
        torch.set_num_threads(int(args.num_threads))

    cfg_infer = OmegaConf.load(args.config)
    network_config_path = Path(args.network_config or cfg_infer.network.config).expanduser()
    checkpoint_path = Path(args.checkpoint or cfg_infer.network.checkpoint).expanduser()
    noisy_dir = Path(args.noisy_dir or cfg_infer.test_dataset.noisy_dir).expanduser()
    clean_dir = Path(args.clean_dir or cfg_infer.test_dataset.clean_dir).expanduser()

    if not network_config_path.exists():
        raise FileNotFoundError(f"Network config not found: {network_config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy dir not found: {noisy_dir}")
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean dir not found: {clean_dir}")

    cfg_network = OmegaConf.load(str(network_config_path))
    network_cfg_raw = OmegaConf.to_container(cfg_network["network_config"], resolve=True)
    network_cfg, dropped_cfg_keys = sanitize_network_cfg_for_model(network_cfg_raw)
    if not isinstance(network_cfg, dict):
        raise RuntimeError("network_config must resolve to a mapping.")
    if dropped_cfg_keys:
        print(
            "Warning: dropped unsupported network_config keys for current GTCRN "
            f"implementation: {dropped_cfg_keys}"
        )

    qat_cfg = cfg_network.get("qat", {})
    backend = str(args.backend or qat_cfg.get("backend", "qnnpack")).strip().lower()
    quantize_deconv_cfg = (
        bool(args.quantize_deconv)
        if args.quantize_deconv is not None
        else bool(qat_cfg.get("quantize_deconv", True))
    )
    per_channel_weights_cfg = (
        bool(args.per_channel_weights)
        if args.per_channel_weights is not None
        else bool(qat_cfg.get("per_channel_weights", True))
    )
    quantize_linear_cfg = (
        bool(args.quantize_linear)
        if args.quantize_linear is not None
        else bool(qat_cfg.get("quantize_linear", False))
    )
    quantize_gru_cfg = (
        bool(args.quantize_gru)
        if args.quantize_gru is not None
        else bool(qat_cfg.get("qat_gru", True))
    )
    dynamic_quantize_gru_cfg = (
        bool(args.dynamic_quantize_gru)
        if args.dynamic_quantize_gru is not None
        else bool(qat_cfg.get("dynamic_quantize_gru", True))
    )
    static_quantize_gru_cfg = (
        bool(args.static_quantize_gru)
        if args.static_quantize_gru is not None
        else bool(qat_cfg.get("static_quantize_gru", False))
    )

    ptq_target = str(args.ptq_target).strip().lower()
    if ptq_target not in {"all", "conv", "gru"}:
        raise ValueError(f"Unsupported --ptq-target: {args.ptq_target}")
    quantize_conv = ptq_target in {"all", "conv"}
    quantize_deconv = quantize_deconv_cfg if quantize_conv else False
    per_channel_weights = per_channel_weights_cfg
    # Linear quantization is controlled independently from conv/deconv and GRU target selection.
    quantize_linear = quantize_linear_cfg
    if ptq_target == "all":
        quantize_gru = quantize_gru_cfg
        dynamic_quantize_gru = dynamic_quantize_gru_cfg
        static_quantize_gru = static_quantize_gru_cfg
    elif ptq_target == "conv":
        quantize_gru = False
        dynamic_quantize_gru = False
        static_quantize_gru = False
    else:
        quantize_gru = True if args.quantize_gru is None else bool(args.quantize_gru)
        dynamic_quantize_gru = (
            True if args.dynamic_quantize_gru is None else bool(args.dynamic_quantize_gru)
        )
        static_quantize_gru = (
            static_quantize_gru_cfg if args.static_quantize_gru is None else bool(args.static_quantize_gru)
        )

    if dynamic_quantize_gru and static_quantize_gru:
        raise ValueError(
            "dynamic_quantize_gru and static_quantize_gru are mutually exclusive."
        )

    bit_widths = sorted(set(parse_csv_ints(args.bit_widths)))
    frac_bits_override = None if normalize_mode_flag(args.frac_bits) in {"all", "none", ""} else parse_csv_ints(args.frac_bits)
    frac_bits_by_bit = build_frac_bits_by_bit(bit_widths, frac_bits_override)
    formats = build_qmn_formats(bit_widths, frac_bits_override)
    metrics = tuple(m.strip().lower() for m in args.metrics.split(",") if m.strip())
    for metric_name in metrics:
        if metric_name not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric '{metric_name}'. Supported: {SUPPORTED_METRICS}")

    all_pairs = collect_wav_pairs(noisy_dir, clean_dir)
    calib_pairs, eval_pairs = split_calib_eval_pairs(all_pairs, args.calib_count, args.eval_count)
    state_dict = load_checkpoint_state(checkpoint_path)

    out_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path("experiments") / "ptq_qmn_sweep" / f"run_{datetime.now().strftime('%Y-%m-%d-%Hh%Mm')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PTQ Qm.n sweep config:")
    print(f"  config:                 {args.config}")
    print(f"  sweep_config:           {args.sweep_config}")
    print(f"  network_config:         {network_config_path}")
    print(f"  checkpoint:             {checkpoint_path}")
    print(f"  backend:                {backend}")
    print(f"  ptq_target:             {ptq_target}")
    print(f"  quantize_conv:          {quantize_conv}")
    print(f"  quantize_deconv:        {quantize_deconv}")
    print(f"  per_channel_weights:    {per_channel_weights}")
    print(f"  quantize_linear:        {quantize_linear}")
    print(f"  quantize_gru:           {quantize_gru}")
    print(f"  dynamic_quantize_gru:   {dynamic_quantize_gru}")
    print(f"  static_quantize_gru:    {static_quantize_gru}")
    print(f"  sweep_mode:             {args.sweep_mode}")
    print(f"  bit_widths:             {bit_widths}")
    print(f"  formats:                {len(formats)}")
    print(f"  calib_files:            {len(calib_pairs)}")
    print(f"  eval_files:             {len(eval_pairs)}")
    print(f"  metrics:                {metrics}")
    print(f"  expected_sample_rate:   {args.sample_rate}")
    print(f"  int8_eval_mode:         {args.int8_eval_mode}")
    print(f"  include_baseline:       {args.include_baseline}")
    print(f"  print_calib_offsets:    {args.print_calib_offsets} ({args.calib_offsets_scope})")
    if args.sweep_mode == "per-layer":
        print(f"  per_layer_scope:        {args.per_layer_scope}")
        print(f"  per_layer_base_n:       {args.per_layer_base_frac_bits}")
        print(f"  per_layer_max_groups:   {args.per_layer_max_groups}")
        print(f"  per_layer_include_base: {args.per_layer_include_base}")
    print(f"  output_dir:             {out_dir}")

    per_layer_groups_by_bits: dict[int, dict[str, object]] = {}
    if args.sweep_mode == "per-layer" or args.list_layer_groups:
        for bits in bit_widths:
            n_values = frac_bits_by_bit[bits]
            base_n = args.per_layer_base_frac_bits
            if base_n is None:
                base_n = int(n_values[0])
            if base_n < 0 or base_n >= bits:
                raise ValueError(f"per-layer base frac bits must be in [0, {bits - 1}], got {base_n} for {bits}-bit.")

            probe_model = build_prepared_model(
                network_cfg=network_cfg,
                state_dict=state_dict,
                device=torch.device("cpu"),
                backend=backend,
                quantize_conv=quantize_conv,
                quantize_deconv=quantize_deconv,
                per_channel_weights=per_channel_weights,
                quantize_linear=quantize_linear,
                quantize_gru=quantize_gru,
                frac_bits=int(base_n),
            )
            _ = set_fake_quant_bit_width(probe_model, bits)
            group_map = collect_fake_quant_groups(probe_model, scope=args.per_layer_scope)
            group_names = list(group_map.keys())
            if args.per_layer_max_groups is not None:
                if args.per_layer_max_groups <= 0:
                    raise ValueError("per_layer_max_groups must be > 0.")
                group_names = group_names[: int(args.per_layer_max_groups)]
            if not group_names:
                raise RuntimeError(
                    f"No fake-quant groups found for per-layer scope='{args.per_layer_scope}' "
                    f"with bit-width={bits}."
                )
            per_layer_groups_by_bits[bits] = {
                "base_n": int(base_n),
                "groups": group_names,
            }
            print(f"Discovered {len(group_names)} per-layer groups for {bits}-bit sweep.")
            if args.list_layer_groups:
                for idx, group_name in enumerate(group_names, start=1):
                    print(f"  [{idx:03d}] {group_name}")
            del probe_model

        if args.list_layer_groups:
            print("Listed per-layer groups only. Exiting.")
            return

    sweep_cases: list[dict[str, object]] = []
    if args.sweep_mode == "global":
        for bits in bit_widths:
            for n in frac_bits_by_bit[bits]:
                sweep_cases.append(
                    {
                        "bits": int(bits),
                        "base_n": int(n),
                        "target_n": int(n),
                        "target_group": None,
                    }
                )
    else:
        for bits in bit_widths:
            group_info = per_layer_groups_by_bits[bits]
            base_n = int(group_info["base_n"])
            groups = list(group_info["groups"])
            n_values = list(frac_bits_by_bit[bits])
            candidate_n_values = n_values if args.per_layer_include_base else [n for n in n_values if n != base_n]
            if not candidate_n_values:
                candidate_n_values = [base_n]
            for group_name in groups:
                for n in candidate_n_values:
                    sweep_cases.append(
                        {
                            "bits": int(bits),
                            "base_n": int(base_n),
                            "target_n": int(n),
                            "target_group": str(group_name),
                        }
                    )

    print(f"  total_sweep_cases:      {len(sweep_cases)}")

    rows: list[dict[str, object]] = []
    if args.include_baseline:
        print("")
        print("[baseline] Evaluating float (no quantization)")
        baseline_model = build_float_model(
            network_cfg=network_cfg,
            state_dict=state_dict,
            device=torch.device("cpu"),
        )
        baseline_summary = evaluate_model(
            baseline_model,
            eval_pairs=eval_pairs,
            device=torch.device("cpu"),
            metrics=metrics,
            expected_sample_rate=args.sample_rate,
        )
        baseline_row: dict[str, object] = {
            "sweep_mode": "baseline",
            "bits": "float",
            "q_format": "FP32",
            "base_q_format": "FP32",
            "m": "",
            "n": "",
            "base_n": "",
            "target_group": "__none__",
            "eval_mode": "float",
            "calib_files": 0,
            "calib_seconds": 0.0,
        }
        baseline_row.update(baseline_summary)
        rows.append(baseline_row)
        baseline_metric_str = ", ".join(
            f"{name}={float(baseline_row.get(name, float('nan'))):.4f}"
            for name in metrics
            if name in baseline_row and not math.isnan(float(baseline_row[name]))
        )
        print(
            f"  Baseline: {baseline_metric_str}, "
            f"rtf={float(baseline_row.get('rtf', float('nan'))):.4f}"
        )
        del baseline_model

    for index, case in enumerate(sweep_cases, start=1):
        bits = int(case["bits"])
        base_n = int(case["base_n"])
        target_n = int(case["target_n"])
        target_group = case["target_group"]
        m = bits - 1 - target_n
        base_m = bits - 1 - base_n
        q_label = f"Q{m}.{target_n}"
        base_q_label = f"Q{base_m}.{base_n}"

        print("")
        if target_group is None:
            print(f"[{index}/{len(sweep_cases)}] Evaluating global {q_label} ({bits}-bit)")
        else:
            print(
                f"[{index}/{len(sweep_cases)}] Evaluating per-layer {q_label} on '{target_group}' "
                f"(base={base_q_label}, {bits}-bit)"
            )

        # In per-layer mode we initialize all modules with base_n then overwrite groups.
        model_init_frac_bits = base_n if target_group is not None else target_n
        base_model = build_prepared_model(
            network_cfg=network_cfg,
            state_dict=state_dict,
            device=torch.device("cpu"),
            backend=backend,
            quantize_conv=quantize_conv,
            quantize_deconv=quantize_deconv,
            per_channel_weights=per_channel_weights,
            quantize_linear=quantize_linear,
            quantize_gru=quantize_gru,
            frac_bits=model_init_frac_bits,
        )

        fake_quant_count = set_fake_quant_bit_width(base_model, bits)
        print(f"  Updated fake-quant bit ranges on {fake_quant_count} modules.")

        if target_group is not None:
            group_names = list(per_layer_groups_by_bits[bits]["groups"])
            group_frac_bits = {name: base_n for name in group_names}
            group_frac_bits[str(target_group)] = target_n
            updated_fake_quants = apply_group_frac_bits(base_model, group_frac_bits)
            print(
                f"  Applied per-layer frac-bits to {updated_fake_quants} fake-quant modules "
                f"(scope groups={len(group_names)})."
            )

        calib_start = time.time()
        run_calibration(
            base_model,
            calib_pairs=calib_pairs,
            device=torch.device("cpu"),
            expected_sample_rate=args.sample_rate,
        )
        calib_elapsed = time.time() - calib_start
        print(f"  Calibration done in {calib_elapsed:.2f}s.")
        if args.print_calib_offsets:
            scope = str(args.calib_offsets_scope).strip().lower()
            if scope == "auto":
                scope = "gru" if ptq_target == "gru" else "fakequant"
            print_calibration_offsets(base_model, scope=scope)

        eval_mode = "fake"
        eval_model = base_model
        dynamic_gru_n = target_n
        if target_group is not None:
            dynamic_gru_n = base_n
            lname = str(target_group).lower()
            if ("gru" in lname) or ("rnn" in lname):
                dynamic_gru_n = target_n

        if bits == 8 and args.int8_eval_mode == "convert":
            try:
                eval_model = base_model.convert_qat(
                    inplace=False,
                    dynamic_quantize_gru=dynamic_quantize_gru,
                    static_quantize_gru=static_quantize_gru,
                    dynamic_gru_scale_constraint_mode="fixed",
                    dynamic_gru_scale_constraint_frac_bits=dynamic_gru_n,
                    dynamic_gru_scale_constraint_pow2_rounding="nearest",
                )
                eval_mode = "convert"
            except Exception as exc:
                print(f"  Warning: int8 convert failed ({type(exc).__name__}: {exc}). Falling back to fake eval.")
                eval_model = base_model
                eval_mode = "fake_fallback"

        metrics_summary = evaluate_model(
            eval_model,
            eval_pairs=eval_pairs,
            device=torch.device("cpu"),
            metrics=metrics,
            expected_sample_rate=args.sample_rate,
        )

        row: dict[str, object] = {
            "sweep_mode": args.sweep_mode,
            "bits": bits,
            "q_format": q_label,
            "base_q_format": base_q_label,
            "m": m,
            "n": target_n,
            "base_n": base_n,
            "target_group": "__all__" if target_group is None else str(target_group),
            "eval_mode": eval_mode,
            "calib_files": len(calib_pairs),
            "calib_seconds": calib_elapsed,
        }
        row.update(metrics_summary)
        rows.append(row)

        metric_str = ", ".join(
            f"{name}={float(row.get(name, float('nan'))):.4f}"
            for name in metrics
            if name in row and not math.isnan(float(row[name]))
        )
        print(f"  Result: {metric_str}, rtf={float(row.get('rtf', float('nan'))):.4f}")

        del eval_model
        del base_model

    run_meta = {
        "config": str(args.config),
        "sweep_config": (str(args.sweep_config) if args.sweep_config is not None else None),
        "network_config": str(network_config_path),
        "checkpoint": str(checkpoint_path),
        "noisy_dir": str(noisy_dir),
        "clean_dir": str(clean_dir),
        "backend": backend,
        "ptq_target": ptq_target,
        "quantize_conv": quantize_conv,
        "quantize_deconv": quantize_deconv,
        "per_channel_weights": per_channel_weights,
        "quantize_linear": quantize_linear,
        "quantize_gru": quantize_gru,
        "dynamic_quantize_gru": dynamic_quantize_gru,
        "static_quantize_gru": static_quantize_gru,
        "sweep_mode": args.sweep_mode,
        "bit_widths": bit_widths,
        "frac_bits_by_bit": frac_bits_by_bit,
        "formats": formats,
        "per_layer_groups_by_bits": per_layer_groups_by_bits,
        "per_layer_scope": args.per_layer_scope,
        "per_layer_base_frac_bits": args.per_layer_base_frac_bits,
        "per_layer_max_groups": args.per_layer_max_groups,
        "per_layer_include_base": args.per_layer_include_base,
        "metrics": list(metrics),
        "calib_count": len(calib_pairs),
        "eval_count": len(eval_pairs),
        "include_baseline": bool(args.include_baseline),
        "sample_rate": args.sample_rate,
        "int8_eval_mode": args.int8_eval_mode,
        "print_calib_offsets": bool(args.print_calib_offsets),
        "calib_offsets_scope": str(args.calib_offsets_scope),
        "timestamp": datetime.now().isoformat(),
    }
    save_results(rows, out_dir, run_meta)
    print_ranked_summary(rows, metrics)


if __name__ == "__main__":
    def build_arg_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Sweep PTQ Qm.n formats for GTCRN.")
        parser.add_argument("-C", "--config", default="configs/cfg_infer.yaml")
        parser.add_argument(
            "--sweep-config",
            default=None,
            help=(
                "YAML file with defaults for PTQ sweep arguments. "
                "CLI flags override values from this file."
            ),
        )
        parser.add_argument("--network-config", default=None, help="Override network config path.")
        parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
        parser.add_argument("--noisy-dir", default=None, help="Override noisy wav directory.")
        parser.add_argument("--clean-dir", default=None, help="Override clean wav directory.")
        parser.add_argument("--output-dir", default=None, help="Output directory for CSV/JSON.")
        parser.add_argument("--sample-rate", type=int, default=16000, help="Expected sample rate. Use -1 to disable check.")
        parser.add_argument("--bit-widths", default="8,16", help="Comma-separated bit widths, e.g. '8,16'.")
        parser.add_argument(
            "--frac-bits",
            default="all",
            help="Comma-separated n values for Qm.n (all if omitted). Example: '3,4,5'.",
        )
        parser.add_argument("--calib-count", type=int, default=64, help="Number of files used for calibration.")
        parser.add_argument(
            "--eval-count",
            type=int,
            default=200,
            help="Number of files for evaluation after calibration split. Use -1 for all remaining.",
        )
        parser.add_argument("--metrics", default="pesq,estoi,sisnr,sdr", help="Comma-separated metrics.")
        parser.add_argument(
            "--ptq-target",
            default="all",
            choices=("all", "conv", "gru"),
            help=(
                "Apply PTQ to all quantizable blocks, only conv/deconv, or only GRU. "
                "Linear layers are controlled independently via --quantize-linear."
            ),
        )
        parser.add_argument(
            "--include-baseline",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run and report a float FP32 baseline before quantized sweeps.",
        )
        parser.add_argument(
            "--int8-eval-mode",
            default="convert",
            choices=("convert", "fake"),
            help="How to evaluate int8 formats: real converted model or fake-quant emulation.",
        )
        parser.add_argument(
            "--sweep-mode",
            default="global",
            choices=("global", "per-layer"),
            help="global: same n for all layers. per-layer: one-layer-at-a-time sweep with a global base n.",
        )
        parser.add_argument(
            "--per-layer-scope",
            default="all",
            choices=("all", "conv", "gru"),
            help="Layer scope used when --sweep-mode per-layer.",
        )
        parser.add_argument(
            "--per-layer-base-frac-bits",
            type=int,
            default=None,
            help="Base global n used in per-layer mode (default: first n from --frac-bits).",
        )
        parser.add_argument(
            "--per-layer-max-groups",
            type=int,
            default=None,
            help="Optional limit on number of layer groups swept in per-layer mode.",
        )
        parser.add_argument(
            "--per-layer-include-base",
            action="store_true",
            help="Also evaluate n=base_n for each target group in per-layer mode.",
        )
        parser.add_argument(
            "--list-layer-groups",
            action="store_true",
            help="Print discovered per-layer fake-quant groups and exit.",
        )
        parser.add_argument(
            "--print-calib-offsets",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Print calibration-derived zero_points (offsets) after each calibration pass.",
        )
        parser.add_argument(
            "--calib-offsets-scope",
            default="auto",
            choices=("auto", "gru", "fakequant"),
            help="Which offsets to print when --print-calib-offsets is enabled.",
        )
        parser.add_argument("--backend", default=None, help="Override quant backend (qnnpack/fbgemm).")
        parser.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads value.")
        parser.add_argument(
            "--quantize-deconv",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable deconv quantization.",
        )
        parser.add_argument(
            "--per-channel-weights",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable per-channel weight quantization.",
        )
        parser.add_argument(
            "--quantize-linear",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable linear layer quantization.",
        )
        parser.add_argument(
            "--quantize-gru",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable GRU fake-quant modules for calibration/emulation.",
        )
        parser.add_argument(
            "--dynamic-quantize-gru",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable GRU dynamic quantization during int8 conversion.",
        )
        parser.add_argument(
            "--static-quantize-gru",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable/disable GRU static Q/DQ emulation during int8 conversion.",
        )
        return parser

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--sweep-config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = build_arg_parser()
    if pre_args.sweep_config:
        valid_keys = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", argparse.SUPPRESS}
        }
        parser.set_defaults(
            **load_sweep_arg_defaults(
                pre_args.sweep_config,
                valid_keys=valid_keys,
            )
        )

    args = parser.parse_args()
    if args.sample_rate is not None and args.sample_rate < 0:
        args.sample_rate = None
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main(args)
