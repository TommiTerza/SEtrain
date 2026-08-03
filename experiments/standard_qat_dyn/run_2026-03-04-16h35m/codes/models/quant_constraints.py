from __future__ import annotations

from typing import Optional

import torch
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    check_min_max_valid,
)


def _to_mode(value: object) -> str:
    if value is None:
        return "none"
    return str(value).strip().lower()


def _constrain_scale_pow2(scale: torch.Tensor, rounding: str) -> torch.Tensor:
    rounding = str(rounding).strip().lower() if rounding is not None else "nearest"
    if rounding in {"", "none", "null"}:
        rounding = "nearest"
    log2 = torch.log2(scale)
    if rounding in {"nearest", "round"}:
        exponent = torch.round(log2)
    elif rounding == "floor":
        exponent = torch.floor(log2)
    elif rounding == "ceil":
        exponent = torch.ceil(log2)
    else:
        raise ValueError(f"Unsupported pow2 rounding mode: {rounding} (expected nearest|floor|ceil)")
    return torch.pow(torch.tensor(2.0, device=scale.device), exponent)


def _constrain_scale_fixed(scale: torch.Tensor, frac_bits: int) -> torch.Tensor:
    frac_bits = int(frac_bits)
    fixed = float(2.0 ** (-frac_bits))
    return torch.full_like(scale, fixed, dtype=torch.float32)


class ConstrainedMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
    """
    MovingAverageMinMaxObserver with optional scale constraints.

    This is useful when you want QAT scales to match fixed-point formats, e.g.
    power-of-two scales (shift-friendly) or fixed 2^-n steps (Qm.n-style).
    """

    def __init__(
        self,
        *args,
        scale_constraint_mode: str = "none",
        scale_constraint_frac_bits: Optional[int] = None,
        scale_constraint_pow2_rounding: str = "nearest",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_constraint_mode = _to_mode(scale_constraint_mode)
        self.scale_constraint_frac_bits = scale_constraint_frac_bits
        self.scale_constraint_pow2_rounding = str(scale_constraint_pow2_rounding).strip().lower()

    def _constrain_scale(self, scale: torch.Tensor) -> torch.Tensor:
        mode = self.scale_constraint_mode
        if mode in {"", "none", "off", "false", "0"}:
            return scale
        if mode in {"pow2", "power2", "power_of_two", "power-of-two"}:
            return _constrain_scale_pow2(scale, self.scale_constraint_pow2_rounding)
        if mode in {"fixed", "qmn", "fixed_qmn", "qformat"}:
            if self.scale_constraint_frac_bits is None:
                raise ValueError("scale_constraint_frac_bits is required for scale_constraint_mode='fixed'")
            return _constrain_scale_fixed(scale, int(self.scale_constraint_frac_bits))
        raise ValueError(f"Unsupported scale_constraint_mode: {mode} (expected none|pow2|fixed)")

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        # This is based on UniformQuantizationObserverBase._calculate_qparams,
        # but with a constraint applied to `scale` and a recomputed zero_point
        # for affine qschemes.
        if not check_min_max_valid(self.min_val, self.max_val):
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(self.min_val, torch.zeros_like(self.min_val))
        max_val_pos = torch.max(self.max_val, torch.zeros_like(self.max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if self.qscheme in {torch.per_tensor_symmetric, torch.per_channel_symmetric}:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            scale = self._constrain_scale(scale)
            scale = torch.max(scale, self.eps)
            if self.dtype in {torch.quint8, torch.uint8}:
                if self.has_customized_qrange:
                    zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (self.max_val - self.min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            scale = self._constrain_scale(scale)
            scale = torch.max(scale, self.eps)
            zero_point = -1 * self.min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            scale = self._constrain_scale(scale)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # Keep FakeQuantize buffer shapes consistent with defaults.
        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor([float(zero_point)], dtype=zero_point.dtype, device=device)
        return scale, zero_point


class ConstrainedMovingAveragePerChannelMinMaxObserver(MovingAveragePerChannelMinMaxObserver):
    """
    MovingAveragePerChannelMinMaxObserver with optional scale constraints.
    """

    def __init__(
        self,
        *args,
        scale_constraint_mode: str = "none",
        scale_constraint_frac_bits: Optional[int] = None,
        scale_constraint_pow2_rounding: str = "nearest",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_constraint_mode = _to_mode(scale_constraint_mode)
        self.scale_constraint_frac_bits = scale_constraint_frac_bits
        self.scale_constraint_pow2_rounding = str(scale_constraint_pow2_rounding).strip().lower()

    def _constrain_scale(self, scale: torch.Tensor) -> torch.Tensor:
        mode = self.scale_constraint_mode
        if mode in {"", "none", "off", "false", "0"}:
            return scale
        if mode in {"pow2", "power2", "power_of_two", "power-of-two"}:
            return _constrain_scale_pow2(scale, self.scale_constraint_pow2_rounding)
        if mode in {"fixed", "qmn", "fixed_qmn", "qformat"}:
            if self.scale_constraint_frac_bits is None:
                raise ValueError("scale_constraint_frac_bits is required for scale_constraint_mode='fixed'")
            return _constrain_scale_fixed(scale, int(self.scale_constraint_frac_bits))
        raise ValueError(f"Unsupported scale_constraint_mode: {mode} (expected none|pow2|fixed)")

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Same as UniformQuantizationObserverBase._calculate_qparams, but with constrained scale.
        if not check_min_max_valid(self.min_val, self.max_val):
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor(
                [0], device=self.min_val.device.type
            )

        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(self.min_val, torch.zeros_like(self.min_val))
        max_val_pos = torch.max(self.max_val, torch.zeros_like(self.max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if self.qscheme in {torch.per_tensor_symmetric, torch.per_channel_symmetric}:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
            scale = self._constrain_scale(scale)
            scale = torch.max(scale, self.eps)
            if self.dtype in {torch.quint8, torch.uint8}:
                if self.has_customized_qrange:
                    zero_point = zero_point.new_full(zero_point.size(), (quant_min + quant_max) // 2)
                else:
                    zero_point = zero_point.new_full(zero_point.size(), 128)
        elif self.qscheme == torch.per_channel_affine_float_qparams:
            scale = (self.max_val - self.min_val) / float(quant_max - quant_min)
            scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))
            scale = self._constrain_scale(scale)
            scale = torch.max(scale, self.eps)
            zero_point = -1 * self.min_val / scale
        else:
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            scale = torch.max(scale, self.eps)
            scale = self._constrain_scale(scale)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)
            if self.qscheme == torch.per_channel_affine_float_qparams:
                zero_point = torch.tensor([float(zero_point)], dtype=zero_point.dtype, device=device)
        return scale, zero_point
