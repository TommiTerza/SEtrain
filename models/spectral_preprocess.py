import math
from typing import Any, Dict, Optional

import torch
from torch import nn


class SpectralPreprocessor(nn.Module):
    """Configurable DSP pre-processing on complex STFTs."""

    def __init__(
        self,
        enable: bool = True,
        highpass: Optional[Dict[str, Any]] = None,
        level: Optional[Dict[str, Any]] = None,
        noise_shave: Optional[Dict[str, Any]] = None,
        wiener: Optional[Dict[str, Any]] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.enable = enable
        self.eps = eps
        self.highpass_cfg = highpass or {}
        self.level_cfg = level or {}
        self.noise_shave_cfg = noise_shave or {}
        self.wiener_cfg = wiener or {}

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply the configured operations to a complex STFT."""
        if not self.enable:
            return spec

        if spec.is_complex() is False:
            raise ValueError("SpectralPreprocessor expects a complex-valued tensor")

        # Avoid in-place edits on tensors that might be reused upstream.
        spec = spec.clone()

        mag = spec.abs()

        if self.highpass_cfg.get("enable", False):
            spec = self._apply_highpass(spec)
            mag = spec.abs()

        if self.level_cfg.get("enable", False):
            spec = self._apply_level(spec)
            mag = spec.abs()

        if self.noise_shave_cfg.get("enable", False):
            spec, mag = self._apply_noise_shave(spec, mag)

        if self.wiener_cfg.get("enable", False):
            spec, mag = self._apply_wiener(spec, mag)

        return spec

    def _apply_highpass(self, spec: torch.Tensor) -> torch.Tensor:
        cutoff = int(self.highpass_cfg.get("cutoff_bin", 0))
        attenuation_db = float(self.highpass_cfg.get("attenuation_db", 0.0))
        if cutoff <= 0 or attenuation_db <= 0.0:
            return spec

        cutoff = min(cutoff, spec.shape[-2])
        scale = math.pow(10.0, -attenuation_db / 20.0)
        spec[..., :cutoff, :] *= scale
        return spec

    def _apply_level(self, spec: torch.Tensor) -> torch.Tensor:
        target_rms = float(self.level_cfg.get("target_rms", 0.0))
        if target_rms <= 0.0:
            return spec

        mag_sq = spec.abs().pow(2)
        current_rms = torch.sqrt(mag_sq.mean(dim=(-2, -1)) + self.eps)

        scale = target_rms / (current_rms + self.eps)
        max_gain_db = float(self.level_cfg.get("max_gain_db", 12.0))
        min_gain_db = float(self.level_cfg.get("min_gain_db", -12.0))
        max_gain = math.pow(10.0, max_gain_db / 20.0)
        min_gain = math.pow(10.0, min_gain_db / 20.0)
        scale = torch.clamp(scale, min=min_gain, max=max_gain)

        scale = scale.view(-1, 1, 1)
        return spec * scale

    def _apply_noise_shave(self, spec: torch.Tensor, mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantile = float(self.noise_shave_cfg.get("quantile", 0.1))
        quantile = float(min(max(quantile, 0.0), 1.0))
        if mag.shape[-1] == 0 or quantile <= 0.0:
            return spec, mag

        scale = float(self.noise_shave_cfg.get("scale", 1.0))
        floor_db = float(self.noise_shave_cfg.get("floor_db", -60.0))
        floor_lin = math.pow(10.0, floor_db / 20.0)

        noise_mag = torch.quantile(mag, quantile, dim=-1, keepdim=True)
        shaved_mag = mag - noise_mag * scale
        floor = mag.new_full((1,), floor_lin)
        shaved_mag = torch.maximum(shaved_mag, floor)
        shaved_mag = torch.minimum(shaved_mag, mag)

        ratio = torch.where(mag > self.eps, shaved_mag / (mag + self.eps), torch.zeros_like(mag))
        spec = spec * ratio
        mag = shaved_mag
        return spec, mag

    def _apply_wiener(self, spec: torch.Tensor, mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantile = float(self.wiener_cfg.get("noise_quantile", 0.1))
        quantile = float(min(max(quantile, 0.0), 1.0))
        if mag.shape[-1] == 0 or quantile <= 0.0:
            return spec, mag

        aggressiveness = float(self.wiener_cfg.get("aggressiveness", 1.0))
        aggressiveness = max(aggressiveness, self.eps)
        min_gain = float(self.wiener_cfg.get("min_gain", 0.1))
        min_gain = float(min(max(min_gain, 0.0), 1.0))

        power = mag.pow(2)
        noise_psd = torch.quantile(power, quantile, dim=-1, keepdim=True)
        residual = (power - noise_psd).clamp(min=0.0)
        snr = residual / (noise_psd + self.eps)
        gain = snr / (snr + aggressiveness)
        gain = gain.clamp(min=min_gain, max=1.0)

        spec = spec * gain
        mag = mag * gain
        return spec, mag
