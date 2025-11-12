"""
Lightweight NumPy-only GRU module for experimentation without PyTorch.

The implementation mirrors the standard gating equations:
    u_t = sigmoid(W_xz x_t + W_hz h_{t-1} + b_z)
    r_t = sigmoid(W_xr x_t + W_hr h_{t-1} + b_r)
    n_t = tanh(W_xn x_t + W_hn (r_t * h_{t-1}) + b_n)
    h_t = (1 - u_t) * n_t + u_t * h_{t-1}

Usage
-----
    >>> import numpy as np
    >>> gru = GRU(input_size=4, hidden_size=3, seed=0)
    >>> x = np.random.randn(2, 5, 4)  # (batch, time, features)
    >>> outputs, hidden = gru.forward(x)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GRUWeights:
    """Container to make saving/loading weights straightforward."""

    Wxu: np.ndarray
    Wxr: np.ndarray
    Wxc: np.ndarray
    Whu: np.ndarray
    Whr: np.ndarray
    Whc: np.ndarray
    bz: np.ndarray
    br: np.ndarray
    bc: np.ndarray


class GRU:
    def __init__(self, input_size: int, hidden_size: int, *, dtype=np.float32, seed: int | None = None):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self._init_parameters()

    # ------------------------------------------------------------------ utils
    def _init_parameters(self) -> None:
        limit = 1.0 / math.sqrt(self.hidden_size)
        def mat(shape):
            return self.rng.uniform(-limit, limit, size=shape).astype(self.dtype)
        # Input weights (x -> gate). We follow PyTorch's gate ordering: reset, update, new.
        self.Wxu = mat((self.hidden_size, self.input_size))
        self.Wxr = mat((self.hidden_size, self.input_size))
        self.Wxc = mat((self.hidden_size, self.input_size))

        # Hidden weights (h -> gate)
        self.Whu = mat((self.hidden_size, self.hidden_size))
        self.Whr = mat((self.hidden_size, self.hidden_size))
        self.Whc = mat((self.hidden_size, self.hidden_size))

        # Biases are split the same way as PyTorch: one term for the input affine, one for hidden.
        self.bz = np.zeros((self.hidden_size,), dtype=self.dtype)
        self.br = np.zeros((self.hidden_size,), dtype=self.dtype)
        self.bc = np.zeros((self.hidden_size,), dtype=self.dtype)

    def reset_parameters(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_parameters()

    # ------------------------------------------------------------- serialization
    def state_dict(self) -> GRUWeights:
        return GRUWeights(
            self.Wxu.copy(), self.Wxr.copy(), self.Wxc.copy(),
            self.Whu.copy(), self.Whr.copy(), self.Whc.copy(),
            self.bz.copy(), self.br.copy(), self.bn_input.copy(), self.bn_hidden.copy(),
        )

    def load_state_dict(self, weights: GRUWeights) -> None:
        for name in ("Wxu", "Wxr", "Wxc", "Whu", "Whr", "Whc", "bz", "br", "bc"):
            setattr(self, name, getattr(weights, name).astype(self.dtype, copy=True))

    # ---------------------------------------------------------------- inference
    def forward(self, inputs: np.ndarray, h0: np.ndarray | None = None,
                *, return_sequence: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        inputs:
            Array with shape ``(T, input_size)`` or ``(B, T, input_size)``.
        h0:
            Optional initial hidden state with shape ``(hidden_size,)`` or ``(B, hidden_size)``.
        return_sequence:
            If True, returns both the full output sequence and the final hidden state.
            If False, returns only the final hidden state (still as the first element of the tuple).
        """
        x = np.asarray(inputs, dtype=self.dtype)
        added_batch = False
        if x.ndim == 2:
            x = x[None, ...]
            added_batch = True
        elif x.ndim != 3:
            raise ValueError("inputs must have shape (T, F) or (B, T, F)")

        batch, time, feats = x.shape
        if feats != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {feats}")

        if h0 is None:
            hidden = np.zeros((batch, self.hidden_size), dtype=self.dtype)
        else:
            h_arr = np.asarray(h0, dtype=self.dtype)
            if h_arr.ndim == 1:
                h_arr = h_arr[None, ...]
            if h_arr.shape != (batch, self.hidden_size):
                raise ValueError(f"h0 must have shape ({batch}, {self.hidden_size})")
            hidden = h_arr

        outputs = np.zeros((batch, time, self.hidden_size), dtype=self.dtype)
        for t in range(time):
            # advance all elements in the batch in lock step
            hidden = self._step(x[:, t, :], hidden)
            outputs[:, t, :] = hidden

        if added_batch:
            outputs = outputs[0]
            hidden = hidden[0]

        if return_sequence:
            return outputs, hidden
        return hidden, hidden

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Public single-step helper for manual control."""
        x = np.asarray(x_t, dtype=self.dtype)
        h = np.asarray(h_prev, dtype=self.dtype)
        if x.ndim == 1:
            x = x[None, :]
        if h.ndim == 1:
            h = h[None, :]
        if x.shape[-1] != self.input_size:
            raise ValueError("x_t has wrong feature dimension")
        if h.shape[-1] != self.hidden_size:
            raise ValueError("h_prev has wrong hidden dimension")
        if x.shape[0] != h.shape[0]:
            raise ValueError("batch dimension mismatch between x_t and h_prev")
        new_h = self._step(x, h)
        return new_h[0] if x_t.ndim == 1 else new_h

    # ----------------------------------------------------------------- internals
    def _step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        # Standard GRU gating sequence, matching PyTorch's implementation.
        # .T is the transpose operator.
        u = _sigmoid(x_t @ self.Wxu.T + h_prev @ self.Whu.T + self.bz)
        r = _sigmoid(x_t @ self.Wxr.T + h_prev @ self.Whr.T + self.br)
        c = np.tanh(x_t @ self.Wxc.T + r * (h_prev @ self.Whc.T)+ self.bc)
        return (1.0 - u) * c + u * h_prev


if __name__ == "__main__":
    gru = GRU(4, 3, seed=42)
    sample = np.random.default_rng(0).standard_normal((5, 4))
    seq, last = gru.forward(sample)
    print("sequence shape:", seq.shape)
    print("last hidden:", last)
