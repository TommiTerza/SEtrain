"""
Search blockwise checkpoint combinations that maximize global PESQ.

This script targets experiments produced by train_blockwise_distill.py:
  <experiment_root>/
    blocks/<sanitized_block_name>/checkpoints/{model_XXX.tar,best_model_XXX.tar}
    config.yaml

It performs coordinate-ascent over per-block epoch choices:
  - initialize with each block's best epoch (from final checkpoint summaries when available)
  - repeatedly sweep blocks, testing all candidate epochs for one block at a time
  - keep updates that improve global PESQ on a fixed validation subset

The objective can be evaluated in fake-quant mode ("fakequant") or after int8 conversion ("int8").
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from pesq import pesq
from tqdm import tqdm

from dataloader import DNS3Dataset as Dataset
from models.gtcrn_end2end import GTCRN as Model


SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _extract_model_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format. Expected dict-like checkpoint.")

    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not find model weights in checkpoint (expected 'model' or 'state_dict').")


def _load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = _extract_model_state_dict(checkpoint)

    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    missing_keys, unexpected_keys = model.load_state_dict(cleaned, strict=False)
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is incompatible. Missing keys: {missing_keys[:10]}"
        )

    ignored_prefixes = ("activation_post_process", "weight_fake_quant")
    bad_unexpected = [k for k in unexpected_keys if not any(pfx in k for pfx in ignored_prefixes)]
    if bad_unexpected:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' has unexpected keys: {bad_unexpected[:10]}"
        )


def _load_payload_into_module(module: torch.nn.Module, payload: dict[str, Any], block_name: str) -> None:
    payload_type = payload.get("payload_type", "full_block")
    if payload_type == "gtconv_conv_path":
        missing_keys, unexpected_keys = module.load_state_dict(payload["model"], strict=False)
        bad_missing = [k for k in missing_keys if not k.startswith("tra.")]
        if bad_missing or unexpected_keys:
            raise RuntimeError(
                f"Failed partial load for GTConv block '{block_name}'. "
                f"bad_missing={bad_missing[:10]}, unexpected={unexpected_keys[:10]}"
            )
        return
    module.load_state_dict(payload["model"], strict=True)


def _resolve_device(device_arg: str, objective_mode: str) -> torch.device:
    if objective_mode == "int8":
        return torch.device("cpu")

    arg = str(device_arg).strip().lower()
    if arg in {"cpu", "none", "-1"}:
        return torch.device("cpu")
    if not arg.isdigit():
        raise ValueError("--device must be a single GPU index (e.g. 0) or 'cpu'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(f"cuda:{int(arg)}")


def _gather_block_candidates(experiment_root: Path) -> tuple[list[str], dict[str, list[int]], dict[tuple[str, int], Path]]:
    blocks_root = experiment_root / "blocks"
    if not blocks_root.exists():
        raise FileNotFoundError(f"Missing blocks directory: {blocks_root}")

    epoch_re = re.compile(r"^(?:best_)?model_(\d+)\.tar$")
    by_block_epochs: dict[str, set[int]] = {}
    by_block_epoch_path: dict[tuple[str, int], Path] = {}

    for ckpt_dir in sorted(blocks_root.glob("*/checkpoints")):
        ckpt_files = sorted(ckpt_dir.glob("*.tar"))
        if not ckpt_files:
            continue

        block_name = None
        for f in ckpt_files:
            payload = torch.load(str(f), map_location="cpu")
            block_name = payload.get("block_name")
            if block_name:
                break
        if not block_name:
            raise RuntimeError(f"Could not infer block_name from checkpoints in {ckpt_dir}")

        by_block_epochs.setdefault(block_name, set())
        for f in ckpt_files:
            m = epoch_re.match(f.name)
            if not m:
                continue
            epoch = int(m.group(1))
            by_block_epochs[block_name].add(epoch)
            key = (block_name, epoch)
            # Prefer regular model_XXX when both exist.
            if key not in by_block_epoch_path or f.name.startswith("model_"):
                by_block_epoch_path[key] = f

    if not by_block_epochs:
        raise RuntimeError(f"No candidate checkpoints found under {blocks_root}")

    block_order = sorted(by_block_epochs.keys())

    final_ckpt = experiment_root / "final_model" / "assembled_student_blockwise.tar"
    if final_ckpt.exists():
        obj = torch.load(str(final_ckpt), map_location="cpu")
        trained_blocks = obj.get("trained_blocks")
        if isinstance(trained_blocks, list) and trained_blocks:
            missing = [b for b in trained_blocks if b not in by_block_epochs]
            if missing:
                raise RuntimeError(f"trained_blocks missing in blocks folder: {missing[:10]}")
            block_order = list(trained_blocks)

    by_block_epochs_sorted = {b: sorted(list(es)) for b, es in by_block_epochs.items()}
    return block_order, by_block_epochs_sorted, by_block_epoch_path


def _initial_combo_from_experiment(
    experiment_root: Path,
    block_order: list[str],
    candidate_epochs: dict[str, list[int]],
) -> dict[str, int]:
    combo = {}
    final_ckpt = experiment_root / "final_model" / "assembled_student_blockwise.tar"
    if final_ckpt.exists():
        obj = torch.load(str(final_ckpt), map_location="cpu")
        summaries = obj.get("block_summaries") or {}
        for block in block_order:
            preferred = None
            if block in summaries:
                preferred = summaries[block].get("best_epoch")
            if isinstance(preferred, int) and preferred in candidate_epochs[block]:
                combo[block] = preferred
            else:
                combo[block] = max(candidate_epochs[block])
        return combo

    for block in block_order:
        combo[block] = max(candidate_epochs[block])
    return combo


def _build_eval_subset(config: dict[str, Any], subset_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if ("validation_dataset" not in config) or ("validation_dataloader" not in config):
        raise ValueError("Experiment config must include validation_dataset + validation_dataloader.")

    dataset = Dataset(**config["validation_dataset"])
    n = len(dataset)
    if subset_size <= 0 or subset_size >= n:
        indices = list(range(n))
    else:
        indices = list(range(n))
        random.Random(SEED).shuffle(indices)
        indices = sorted(indices[:subset_size])

    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx in tqdm(indices, desc="Loading validation subset", ncols=120, dynamic_ncols=True):
        noisy, clean = dataset[idx]
        samples.append((torch.from_numpy(noisy), torch.from_numpy(clean)))
    return samples


def _compute_pesq_batch(clean_batch: np.ndarray, enhanced_batch: np.ndarray) -> tuple[float, int]:
    score_sum = 0.0
    score_count = 0
    for clean, enhanced in zip(clean_batch, enhanced_batch):
        try:
            score = float(pesq(16000, clean, enhanced, "wb"))
        except Exception:
            continue
        if math.isfinite(score):
            score_sum += score
            score_count += 1
    return score_sum, score_count


def _evaluate_pesq(
    model: torch.nn.Module,
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0

    with torch.inference_mode():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            noisy = torch.stack([x[0] for x in batch], dim=0).to(device)
            clean = torch.stack([x[1] for x in batch], dim=0).cpu().numpy()

            enhanced = model(noisy).detach().cpu().numpy()
            score_sum, score_count = _compute_pesq_batch(clean, enhanced)
            total += score_sum
            count += score_count

    if count == 0:
        return float("-inf")
    return total / count


class CombinationSearcher:
    def __init__(
        self,
        experiment_root: Path,
        config: dict[str, Any],
        block_order: list[str],
        candidate_epochs: dict[str, list[int]],
        checkpoint_paths: dict[tuple[str, int], Path],
        samples: list[tuple[torch.Tensor, torch.Tensor]],
        objective_mode: str,
        device: torch.device,
        batch_size: int,
    ):
        self.experiment_root = experiment_root
        self.config = config
        self.block_order = block_order
        self.candidate_epochs = candidate_epochs
        self.checkpoint_paths = checkpoint_paths
        self.samples = samples
        self.objective_mode = objective_mode
        self.device = device
        self.batch_size = batch_size

        self.payload_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self.score_cache: dict[tuple[int, ...], float] = {}

        teacher_ckpt = Path(str(config["distillation"]["teacher_checkpoint"])).expanduser()
        if not teacher_ckpt.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

        self.model = Model(**config["network_config"]).to(self.device)
        _load_model_weights(self.model, teacher_ckpt, self.device)
        self.qat_enabled = bool(config.get("qat", {}).get("enabled", False))
        if self.qat_enabled:
            self.model.prepare_qat(
                backend=config.get("qat", {}).get("backend", "fbgemm"),
                quantize_deconv=bool(config.get("qat", {}).get("quantize_deconv", False)),
            )
        self.base_state = deepcopy(self.model.state_dict())

    def _load_payload(self, block_name: str, epoch: int) -> dict[str, Any]:
        key = (block_name, epoch)
        payload = self.payload_cache.get(key)
        if payload is not None:
            return payload
        path = self.checkpoint_paths[key]
        payload = torch.load(str(path), map_location="cpu")
        self.payload_cache[key] = payload
        return payload

    def _assemble_model(self, combo: dict[str, int]) -> None:
        self.model.load_state_dict(self.base_state, strict=True)
        modules = dict(self.model.named_modules())
        for block_name in self.block_order:
            payload = self._load_payload(block_name, combo[block_name])
            if block_name not in modules:
                raise RuntimeError(f"Block '{block_name}' missing in model during assembly.")
            _load_payload_into_module(modules[block_name], payload, block_name)

    def evaluate_combo(self, combo: dict[str, int]) -> float:
        key = tuple(combo[b] for b in self.block_order)
        if key in self.score_cache:
            return self.score_cache[key]

        self._assemble_model(combo)
        if self.objective_mode == "int8":
            model_for_eval = self.model.cpu().convert_qat(inplace=False)
            score = _evaluate_pesq(
                model=model_for_eval,
                samples=self.samples,
                batch_size=self.batch_size,
                device=torch.device("cpu"),
            )
        else:
            score = _evaluate_pesq(
                model=self.model,
                samples=self.samples,
                batch_size=self.batch_size,
                device=self.device,
            )

        self.score_cache[key] = score
        return score

    def save_assembled_checkpoint(self, combo: dict[str, int], score: float, output_path: Path) -> None:
        self._assemble_model(combo)
        ckpt = {
            "model": self.model.state_dict(),
            "trained_blocks": self.block_order,
            "selected_epochs": combo,
            "objective": {
                "name": "pesq",
                "mode": self.objective_mode,
                "score": score,
                "subset_size": len(self.samples),
            },
            "assembled_from_teacher_base": True,
            "source_experiment": str(self.experiment_root),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, str(output_path))


def _apply_block_limit(block_order: list[str], max_blocks: int) -> list[str]:
    if max_blocks <= 0:
        return block_order
    return block_order[: min(max_blocks, len(block_order))]


def _filter_epochs(
    epochs: list[int],
    epoch_min: int,
    epoch_max: int,
    epoch_step: int,
    epoch_limit: int,
) -> list[int]:
    out = list(epochs)
    if epoch_min > 0:
        out = [e for e in out if e >= epoch_min]
    if epoch_max > 0:
        out = [e for e in out if e <= epoch_max]
    if epoch_step > 1:
        out = [e for e in out if (e % epoch_step == 0)]

    if epoch_limit > 0 and len(out) > epoch_limit:
        idx = np.linspace(0, len(out) - 1, num=epoch_limit, dtype=int)
        out = [out[i] for i in idx]
    return sorted(set(out))


def main(args: argparse.Namespace) -> None:
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    config_path = experiment_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    config = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    block_order, candidate_epochs, checkpoint_paths = _gather_block_candidates(experiment_root)
    block_order = _apply_block_limit(block_order, args.max_blocks)
    candidate_epochs = {b: candidate_epochs[b] for b in block_order}
    for block_name in block_order:
        filtered = _filter_epochs(
            epochs=candidate_epochs[block_name],
            epoch_min=args.epoch_min,
            epoch_max=args.epoch_max,
            epoch_step=args.epoch_step,
            epoch_limit=args.epoch_limit,
        )
        if not filtered:
            raise RuntimeError(
                f"Filtering removed all candidates for block '{block_name}'. "
                "Adjust --epoch_min/--epoch_max/--epoch_step/--epoch_limit."
            )
        candidate_epochs[block_name] = filtered

    device = _resolve_device(args.device, args.objective_mode)
    samples = _build_eval_subset(config, subset_size=args.subset_size)

    searcher = CombinationSearcher(
        experiment_root=experiment_root,
        config=config,
        block_order=block_order,
        candidate_epochs=candidate_epochs,
        checkpoint_paths=checkpoint_paths,
        samples=samples,
        objective_mode=args.objective_mode,
        device=device,
        batch_size=args.batch_size,
    )

    combo = _initial_combo_from_experiment(experiment_root, block_order, candidate_epochs)
    for block_name in block_order:
        if combo[block_name] not in candidate_epochs[block_name]:
            # Keep a deterministic fallback if the preferred epoch was filtered out.
            combo[block_name] = candidate_epochs[block_name][-1]
    score = searcher.evaluate_combo(combo)
    print(f"Initial combo score: {score:.6f}")
    print("Initial epochs:", combo)

    for pass_idx in range(1, args.max_passes + 1):
        improved = False
        print(f"\n=== Pass {pass_idx}/{args.max_passes} ===")
        for block_name in block_order:
            current_epoch = combo[block_name]
            best_epoch = current_epoch
            best_score = score
            print(f"\nSweeping {block_name} (current epoch={current_epoch}, score={score:.6f})")

            for epoch in candidate_epochs[block_name]:
                if epoch == current_epoch:
                    continue
                trial = dict(combo)
                trial[block_name] = epoch
                trial_score = searcher.evaluate_combo(trial)
                print(f"  epoch {epoch:>3d} -> PESQ {trial_score:.6f}")
                if trial_score > best_score:
                    best_score = trial_score
                    best_epoch = epoch

            if best_epoch != current_epoch:
                combo[block_name] = best_epoch
                score = best_score
                improved = True
                print(
                    f"  Updated {block_name}: epoch {current_epoch} -> {best_epoch}, "
                    f"new score={score:.6f}"
                )
            else:
                print("  No improvement for this block.")

        if not improved:
            print(f"\nNo improvements in pass {pass_idx}; stopping early.")
            break

    stamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
    output_dir = experiment_root / f"epoch_search_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_ckpt = output_dir / "assembled_student_blockwise_pesq_search.tar"
    searcher.save_assembled_checkpoint(combo, score, out_ckpt)

    ranked = sorted(searcher.score_cache.items(), key=lambda kv: kv[1], reverse=True)
    top_combinations = []
    for key, key_score in ranked[: args.topk_summary]:
        top_combinations.append(
            {
                "score": key_score,
                "epochs": {block_name: int(epoch) for block_name, epoch in zip(block_order, key)},
            }
        )

    summary = {
        "experiment_root": str(experiment_root),
        "objective_mode": args.objective_mode,
        "device": str(device),
        "subset_size": len(samples),
        "batch_size": args.batch_size,
        "max_passes": args.max_passes,
        "blocks": block_order,
        "selected_epochs": combo,
        "final_score": score,
        "num_evaluated_combinations": len(searcher.score_cache),
        "top_combinations": top_combinations,
        "output_checkpoint": str(out_ckpt),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n=== Search complete ===")
    print(f"Final PESQ: {score:.6f}")
    print(f"Selected epochs: {combo}")
    print(f"Evaluated combinations: {len(searcher.score_cache)}")
    print(f"Saved assembled checkpoint: {out_ckpt}")
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_root",
        required=True,
        help="Blockwise experiment directory, e.g. experiments_independent_blocks_2026-02-21-09h46m",
    )
    parser.add_argument(
        "--objective_mode",
        default="int8",
        choices=["fakequant", "int8"],
        help="Evaluate PESQ in fake-quant model or int8-converted model.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU index for fakequant mode, or 'cpu'. Ignored for int8 mode.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=96,
        help="Validation subset size for objective. <=0 uses full validation set.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_passes", type=int, default=2)
    parser.add_argument(
        "--epoch_min",
        type=int,
        default=-1,
        help="Minimum candidate epoch (inclusive). <=0 disables.",
    )
    parser.add_argument(
        "--epoch_max",
        type=int,
        default=-1,
        help="Maximum candidate epoch (inclusive). <=0 disables.",
    )
    parser.add_argument(
        "--epoch_step",
        type=int,
        default=2,
        help="Keep only epochs divisible by this value (default 2 keeps all model_XXX in these runs).",
    )
    parser.add_argument(
        "--epoch_limit",
        type=int,
        default=-1,
        help="Optional cap per block after filtering (uniformly sampled). <=0 disables.",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=-1,
        help="Debug option: only optimize the first N blocks in trained_blocks order.",
    )
    parser.add_argument(
        "--topk_summary",
        type=int,
        default=10,
        help="How many top combinations to store in summary.json.",
    )
    cli_args = parser.parse_args()
    main(cli_args)
