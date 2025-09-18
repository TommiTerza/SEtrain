"""Loss functions for time-domain GTCRN training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant signal-to-distortion ratio."""
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    target_energy = torch.sum(target * target, dim=-1, keepdim=True) + eps
    projection = torch.sum(pred * target, dim=-1, keepdim=True) * target / target_energy
    noise = pred - projection

    ratio = torch.sum(projection * projection, dim=-1, keepdim=True) / (
        torch.sum(noise * noise, dim=-1, keepdim=True) + eps
    )
    return 10.0 * torch.log10(ratio + eps)


def si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Backward-compatible alias for SI-SDR."""
    return si_sdr(pred, target, eps)


@dataclass
class STFTConfig:
    n_fft: int
    hop_length: int
    win_length: int
    window: str = "hann_window"


class STFTLoss(nn.Module):
    """Single-resolution STFT loss returning magnitude and complex residuals."""

    def __init__(self, cfg: STFTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        window = getattr(torch, cfg.window)(cfg.win_length)
        self.register_buffer("window", window, persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_spec = torch.stft(
            pred,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window.to(pred.device),
            return_complex=True,
        )
        target_spec = torch.stft(
            target,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=self.window.to(target.device),
            return_complex=True,
        )

        mag_loss = F.l1_loss(pred_spec.abs(), target_spec.abs())
        complex_loss = F.l1_loss(pred_spec.real, target_spec.real) + F.l1_loss(pred_spec.imag, target_spec.imag)
        return mag_loss, complex_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Average magnitude + complex loss across multiple FFT settings."""

    def __init__(self, configs: Sequence[STFTConfig]) -> None:
        super().__init__()
        if not configs:
            raise ValueError("At least one STFT configuration is required")
        self.losses = nn.ModuleList(STFTLoss(cfg) for cfg in configs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mag = 0.0
        complex_val = 0.0
        for loss in self.losses:
            m, c = loss(pred, target)
            mag = mag + m
            complex_val = complex_val + c
        scale = 1.0 / len(self.losses)
        return mag * scale, complex_val * scale


class WaveformLoss(nn.Module):
    """Composite loss: SI-SDR plus multi-resolution STFT terms."""

    def __init__(
        self,
        si_snr_weight: Optional[float] = None,
        si_sdr_weight: Optional[float] = None,
        mag_weight: float = 0.1,
        complex_weight: float = 0.1,
        mrstft: Iterable[dict] | None = None,
    ) -> None:
        super().__init__()
        if si_sdr_weight is None:
            si_sdr_weight = 1.0 if si_snr_weight is None else si_snr_weight
        elif si_snr_weight is not None:
            raise ValueError("Specify only one of si_sdr_weight or si_snr_weight")
        self.si_sdr_weight = si_sdr_weight
        self.mag_weight = mag_weight
        self.complex_weight = complex_weight

        if mrstft is None:
            mrstft = [
                {
                    "n_fft": 2048,
                    "hop_length": 240,
                    "win_length": 1200,
                    "window": "hann_window",
                },
                {
                    "n_fft": 1024,
                    "hop_length": 120,
                    "win_length": 600,
                    "window": "hann_window",
                },
                {
                    "n_fft": 512,
                    "hop_length": 50,
                    "win_length": 240,
                    "window": "hann_window",
                },
            ]
        configs = [cfg if isinstance(cfg, STFTConfig) else STFTConfig(**cfg) for cfg in mrstft]
        self.mrstft = MultiResolutionSTFTLoss(configs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sisdr = -si_sdr(pred, target).mean()
        mag_loss, complex_loss = self.mrstft(pred, target)
        return self.si_sdr_weight * sisdr + self.mag_weight * mag_loss + self.complex_weight * complex_loss


__all__ = ["WaveformLoss", "si_sdr", "si_snr", "MultiResolutionSTFTLoss", "STFTLoss", "STFTConfig"]
