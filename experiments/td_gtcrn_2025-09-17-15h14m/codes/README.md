# TD-GTCRN Speech Enhancement Template

This repository hosts a compact training template for single-channel speech enhancement centred on a **time-domain GTCRN** (Grouped Temporal Convolutional Recurrent Network). The original STFT front/back-end has been replaced with a fully learnable, causal audio codec inspired by TasNet. The GTCRN core (ShuffleNet-style encoder, grouped dual-path RNNs, temporal recurrent attention) remains intact and now operates directly on latent codec channels.

Key features:
- **Waveform codec** – causal Conv1d analysis/synthesis with optional stream-friendly padding.
- **Latent ratio masking** – bounded real-valued masks applied on codec coefficients.
- **Causal normalisation** – replaces 2-D batch norm with cumulative layer norm for low-latency streaming.
- **Waveform-first training** – composite SI-SNR + multi-resolution STFT loss (magnitude + complex residual).
- **Makefile workflow** – helpers for dataset prep, training, inference, and profiling, mirroring the latest `tstnn` UX.

## Repository Layout
- `configs/` – training (`cfg_train.yaml`) and inference (`cfg_infer.yaml`) configs with codec settings.
- `dataloader.py` – VoiceBank-DEMAND loader with on-demand Hugging Face download.
- `loss_factory.py` – waveform loss stack (SI-SNR + MR-STFT).
- `models/gtcrn_end2end.py` – time-domain GTCRN model definition.
- `scripts/prepare_voicebank.py` – CLI wrapper around dataset preparation.
- `scripts/profile_td_gtcrn.py` – lightweight profiling harness (inference or train step).
- `train.py` / `infer.py` – training and inference entry points.
- `Makefile` – convenience targets for the common workflow.

Legacy DNS speech enhancement utilities (`prepare_datasets/`, `evaluation/`, `DNSMOS/`) remain untouched for compatibility.

## Quick Start

```bash
# 1. Download + cache VoiceBank-DEMAND (16 kHz)
make prepare-voicebank

# 2. Train TD-GTCRN (edit configs/cfg_train.yaml as needed)
make train

# 3. Run inference on the validation split
make infer

# 4. Capture profiler traces (requires torch>=2.0)
make profile-infer
make profile-train
```

Variables such as `PYTHON`, `CONFIG`, `INFER_CONFIG`, and `DATA_DIR` can be overridden on the command line, e.g. `make PYTHON=python3.11 train`.

## Model Overview

- **Analysis encoder**: causal Conv1d with kernel `L`, stride `S=L/2` by default, Softplus activation to enforce non-negativity, and padding logic that keeps overlap-add perfect reconstruction.
- **Latent processing**: GTCRN encoder blocks, grouped dual-path GRUs (intra = latent channels, inter = time), and temporal recurrent attention are reused with minimal changes. Batch norm is swapped for cumulative layer norm to stay causal.
- **Masking & synthesis**: the decoder predicts a ratio mask in `[0,1]` (or `[-1,1]` if `mask_activation` is set to `tanh`) applied on latent channels before ConvTranspose1d overlap-add reconstruction.
- **Latency**: algorithmic latency ≈ `(kernel_size - stride)` samples plus any scheduling lookahead. With `kernel_size=64`, `stride=32` @ 16 kHz this is ≈2 ms.

## Losses

`loss_factory.WaveformLoss` combines:
1. **SI-SNR** – primary objective.
2. **Multi-resolution STFT** – averaged magnitude + complex residual losses across three FFT scales (configurable).

Tweak the weights (`si_snr_weight`, `mag_weight`, `complex_weight`) or add additional STFT configurations via `loss.mrstft` in `cfg_train.yaml`.

## Streaming Notes

- The codec stores `kernel_size - stride` samples of left context implicitly through padding; keep this buffer between chunks for seamless streaming.
- GRU state caching can be layered on top by maintaining the dual-path GRNN hidden states outside the `forward` call. The current implementation exposes pure waveform inference; extend it with a streaming wrapper if online deployment is required.

## Evaluation & Metrics

Run `infer.py` to generate enhanced waveforms; it produces `inf.scp` / `ref.scp` files under the configured `enh_folder`. `evaluate.py` can then call DNSMOS or intrusive metrics scripts as before.

## Troubleshooting

- **PyTorch missing** – profiling scripts and training obviously require `torch`. Install `torch>=2.0` that matches your CUDA stack.
- **Dataset download** – the Hugging Face dataset loader honours the cache directory in `datasets/`. Use `HF_DATASETS_CACHE` if you need a custom location.
- **Multi-GPU** – set `-D 0,1,...` when invoking `train.py` directly or edit `DDP.world_size` in the config.

Happy experimenting!
