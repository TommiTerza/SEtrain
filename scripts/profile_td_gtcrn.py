"""Profiler helpers for TD-GTCRN."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import torch
    from torch.profiler import ProfilerActivity, profile
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "PyTorch is required for profiling. Install torch==2.x and retry."
    ) from exc

from models.gtcrn_end2end import GTCRN
from loss_factory import WaveformLoss


def _build_model(args: argparse.Namespace) -> GTCRN:
    codec_cfg = {
        "latent_channels": args.codec_channels,
        "kernel_size": args.codec_kernel,
        "stride": args.codec_stride,
        "activation": args.codec_activation,
        "mask_activation": args.mask_activation,
    }
    return GTCRN(codec=codec_cfg)


def profile_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args).to(device)
    model.eval()

    audio = torch.randn(1, args.sample_length, device=device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model(audio)

    with profile(activities=activities, with_stack=True, record_shapes=True) as prof:
        with torch.no_grad():
            model(audio)
    prof.export_chrome_trace(str(out_path))
    print(f"[td-gtcrn] Inference profile written to {out_path}")


def profile_train_step(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args).to(device)
    loss_fn = WaveformLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    noisy = torch.randn(args.batch_size, args.sample_length, device=device)
    clean = torch.randn_like(noisy)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(noisy)
    loss = loss_fn(output, clean)
    loss.backward()
    optimizer.step()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with profile(activities=activities, with_stack=True, record_shapes=True) as prof:
        output = model(noisy)
        loss = loss_fn(output, clean)
        loss.backward()
        optimizer.step()
    prof.export_chrome_trace(str(out_path))
    print(f"[td-gtcrn] Training profile written to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TD-GTCRN profiler helper")
    parser.add_argument("mode", choices=["inference", "train"], help="Profile inference or a train step")
    parser.add_argument("--sample-length", type=int, default=16000 * 4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/profiles/trace.json"))
    parser.add_argument("--codec-channels", type=int, default=128)
    parser.add_argument("--codec-kernel", type=int, default=64)
    parser.add_argument("--codec-stride", type=int, default=32)
    parser.add_argument("--codec-activation", type=str, default="softplus")
    parser.add_argument("--mask-activation", type=str, default="sigmoid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "inference":
        profile_inference(args)
    else:
        profile_train_step(args)


if __name__ == "__main__":
    main()
