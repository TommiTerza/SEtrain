from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import torch
import torch.nn as nn


__all__ = ["DeltaGRU"]


class DeltaGRU(nn.Module):
    """
    GRU with delta-gated computations.

    For each input feature, if |x(t) - x(t-1)| < threshold_x, its contribution
    to W_ih is skipped (zeroed) for that timestep. The same rule applies to each
    hidden feature with threshold_h for W_hh. Optional logging mirrors the
    gtcrn_end2end logging helpers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        threshold_x: Optional[float] = None,
        threshold_h: Optional[float] = None,
        log_gru_inputs: bool = False,
        log_file_base: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
            **factory_kwargs,
        )
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.threshold_x = threshold_x
        self.threshold_h = threshold_h
        self.log_gru_inputs = log_gru_inputs
        self.log_file_base = log_file_base
        self._log_paths: Optional[dict[str, str]] = None
        self._init_logging()

    def _init_logging(self) -> None:
        if not self.log_gru_inputs or self.log_file_base is None:
            return
        base, ext = os.path.splitext(self.log_file_base)
        ext = ext if ext else ".pkl"
        self._log_paths = {
            "x": f"{base}_x{ext}",
            "h": f"{base}_h{ext}",
        }
        for path in self._log_paths.values():
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D input (B,T,C) or (T,B,C), got {tuple(x.shape)}.")

        self.gru.flatten_parameters()
        time_major = x.transpose(0, 1) if self.batch_first else x  # (T,B,C)
        seq_len, batch_size, _ = time_major.shape

        if h0 is None:
            h0 = self._init_hidden(batch_size, x)

        if self.num_directions == 1:
            outputs, h_last = self._run_direction(time_major, h0, reverse=False)
        else:
            h0_view = h0.reshape(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            fwd_outputs, fwd_hidden = self._run_direction(time_major, h0_view[:, 0], reverse=False)
            bwd_outputs, bwd_hidden = self._run_direction(time_major.flip(0), h0_view[:, 1], reverse=True)
            outputs = torch.cat([fwd_outputs, bwd_outputs], dim=-1)
            h_last = torch.stack([fwd_hidden, bwd_hidden], dim=1).reshape(
                self.num_layers * self.num_directions, batch_size, self.hidden_size
            )

        output = outputs.transpose(0, 1) if self.batch_first else outputs
        return output, h_last

    def _init_hidden(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        device = reference.device
        dtype = reference.dtype
        return torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _threshold_active(value: Optional[float]) -> bool:
        return value is not None and value > 0

    def _run_direction(
        self,
        seq: torch.Tensor,
        h0: torch.Tensor,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, batch_size, _ = seq.shape
        prev_x_actual: Optional[torch.Tensor] = None
        prev_hidden = [h0[layer] for layer in range(self.num_layers)]
        prev_prev_hidden: list[Optional[torch.Tensor]] = [None] * self.num_layers
        # Projections caches per layer for the input passed to that layer
        cache_input_val: list[Optional[torch.Tensor]] = [None] * self.num_layers
        cache_input_proj: list[Optional[torch.Tensor]] = [None] * self.num_layers
        # Projection caches per layer for the hidden value fed to that layer
        cache_hidden_val: list[Optional[torch.Tensor]] = [None] * self.num_layers
        cache_hidden_proj: list[Optional[torch.Tensor]] = [None] * self.num_layers
        outputs: list[torch.Tensor] = []

        logging_active = (
            self.log_gru_inputs
            and self._log_paths is not None
            and not self.training
        )
        log_inputs: list[torch.Tensor] = []

        for t in range(seq_len):
            x_curr_actual = seq[t]
            x_masked = self._mask_input(x_curr_actual, prev_x_actual)
            prev_x_actual = x_curr_actual
            if logging_active:
                log_inputs.append(x_curr_actual.detach())

            layer_input = x_masked
            for layer in range(self.num_layers):
                h_prev_actual = prev_hidden[layer]
                h_prev_prev_actual = prev_prev_hidden[layer]
                h_masked = self._mask_hidden(h_prev_actual, h_prev_prev_actual)
                weight_ih, weight_hh, bias_ih, bias_hh = self._get_gru_params(layer, reverse)
                # Input projection caching per layer
                gi, cache_input_val[layer], cache_input_proj[layer] = self._project_with_cache(
                    layer_input,
                    weight_ih,
                    self.threshold_x,
                    cache_input_val[layer],
                    cache_input_proj[layer],
                )
                # Hidden projection caching per layer (uses masked hidden fed to this layer)
                gh, cache_hidden_val[layer], cache_hidden_proj[layer] = self._project_with_cache(
                    h_masked,
                    weight_hh,
                    self.threshold_h,
                    cache_hidden_val[layer],
                    cache_hidden_proj[layer],
                )
                layer_input = self._gru_cell(
                    layer_input,
                    h_prev_actual,
                    h_masked,
                    gi,
                    gh,
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                )
                # Update state trackers for next timestep
                prev_prev_hidden[layer] = h_prev_actual
                prev_hidden[layer] = layer_input
            outputs.append(layer_input)

        outputs_tensor = torch.stack(outputs, dim=0)
        if reverse:
            outputs_tensor = outputs_tensor.flip(0)
        hidden_tensor = torch.stack(prev_hidden, dim=0)

        if logging_active and self._log_paths is not None:
            payload_x = torch.cat(log_inputs, dim=0).cpu().reshape(-1, log_inputs[0].shape[-1]).numpy() if log_inputs else None
            payload_h = outputs_tensor.detach().cpu()
            if payload_h.dim() == 3:
                payload_h = payload_h.permute(1, 0, 2).reshape(-1, payload_h.shape[-1])
            payload_h_np = payload_h.numpy()
            if payload_x is not None:
                with open(self._log_paths["x"], "wb") as f:
                    pickle.dump(payload_x, f)
            with open(self._log_paths["h"], "wb") as f:
                pickle.dump(payload_h_np, f)

        return outputs_tensor, hidden_tensor

    def _get_gru_params(self, layer: int, reverse: bool):
        suffix = "" if not reverse else "_reverse"
        weight_ih = getattr(self.gru, f"weight_ih_l{layer}{suffix}")
        weight_hh = getattr(self.gru, f"weight_hh_l{layer}{suffix}")
        bias_ih = getattr(self.gru, f"bias_ih_l{layer}{suffix}", None)
        bias_hh = getattr(self.gru, f"bias_hh_l{layer}{suffix}", None)
        return weight_ih, weight_hh, bias_ih, bias_hh

    def _gru_cell(
        self,
        input_t: torch.Tensor,
        hidden_actual: torch.Tensor,
        hidden_masked: torch.Tensor,
        proj_input: torch.Tensor,
        proj_hidden: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: Optional[torch.Tensor],
        bias_hh: Optional[torch.Tensor],
    ) -> torch.Tensor:
        gi = proj_input
        gh = proj_hidden
        if bias_ih is not None:
            gi = gi + bias_ih
        if bias_hh is not None:
            gh = gh + bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + updategate * (hidden_actual - newgate)
        return hy

    def _mask_input(self, current: torch.Tensor, prev_actual: Optional[torch.Tensor]) -> torch.Tensor:
        if prev_actual is None or not self._threshold_active(self.threshold_x):
            return current
        delta = (current - prev_actual).abs()
        reuse_prev = delta < self.threshold_x
        return torch.where(reuse_prev, prev_actual, current)

    def _mask_hidden(
        self,
        prev_actual: torch.Tensor,
        prev_prev_actual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prev_prev_actual is None or not self._threshold_active(self.threshold_h):
            return prev_actual
        delta = (prev_actual - prev_prev_actual).abs()
        reuse_prev_prev = delta < self.threshold_h
        return torch.where(reuse_prev_prev, prev_prev_actual, prev_actual)

    @staticmethod
    def _increment_projection(
        base_proj: torch.Tensor,
        diff: torch.Tensor,
        weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Incrementally update projections for changed features only.
        base_proj: (B, gate_dim)
        diff: (B, F)
        weight: (gate_dim, F)
        mask: (B, F) bool -> True where feature changed enough to recompute
        """
        updated = base_proj.clone()
        B = diff.shape[0]
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            contrib = torch.matmul(diff[b, idx], weight[:, idx].t())
            updated[b] = updated[b] + contrib
        return updated

    def _project_with_cache(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        threshold: Optional[float],
        cache_val: Optional[torch.Tensor],
        cache_proj: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reuse cached projection when deltas are below threshold; otherwise update incrementally.
        Returns (projection, new_cache_val, new_cache_proj).
        """
        if cache_val is None or not self._threshold_active(threshold):
            proj = torch.matmul(value, weight.t())
            return proj, value.detach(), proj.detach()
        diff = value - cache_val
        mask = diff.abs() >= threshold
        if mask.any():
            proj = self._increment_projection(cache_proj, diff, weight, mask)
            return proj, value.detach(), proj.detach()
        else:
            return cache_proj, cache_val, cache_proj
