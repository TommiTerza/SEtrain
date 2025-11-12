#!/usr/bin/env python3
"""
Utility to compare the NumPy GRU (custom_gru.GRU) against torch.nn.GRU.

Example
-------
    python compare_custom_gru.py --input-size 32 --hidden-size 16 --seq-len 40 --batch 4 --seed 123
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from custom_gru import GRU


def _chunk_weights(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split stacked torch weights into (reset, update, new) blocks."""
    chunks = torch.chunk(tensor, 3, dim=0)
    return tuple(chunk.detach().cpu().numpy() for chunk in chunks)  # type: ignore[return-value]


def sync_weights(torch_gru: torch.nn.GRU, custom_gru: GRU) -> None:
    """Copy PyTorch GRU weights into the NumPy GRU instance."""
    if torch_gru.num_layers != 1 or torch_gru.bidirectional:
        raise ValueError("This helper only supports single-layer, unidirectional GRUs.")

    X_r, X_z, X_n = _chunk_weights(torch_gru.weight_ih_l0)
    H_r, H_z, H_n = _chunk_weights(torch_gru.weight_hh_l0)
    b_r_ih, b_z_ih, b_n_ih = _chunk_weights(torch_gru.bias_ih_l0)
    b_r_hh, b_z_hh, b_n_hh = _chunk_weights(torch_gru.bias_hh_l0)

    custom_gru.Wxr = X_r.copy()
    custom_gru.Wxu = X_z.copy()
    custom_gru.Wxc = X_n.copy()
    custom_gru.Whr = H_r.copy()
    custom_gru.Whu = H_z.copy()
    custom_gru.Whc = H_n.copy()
    custom_gru.br = (b_r_ih + b_r_hh).copy()
    custom_gru.bz = (b_z_ih + b_z_hh).copy()
    custom_gru.bc = b_n_ih.copy() + b_n_hh.copy()


def run_comparison(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch_gru = torch.nn.GRU(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        batch_first=True,
        bias=True,
    )
    torch_gru.eval()

    custom = GRU(args.input_size, args.hidden_size, seed=args.seed)
    sync_weights(torch_gru, custom)

    x = torch.randn(args.batch, args.seq_len, args.input_size)
    h0 = torch.randn(1, args.batch, args.hidden_size)

    with torch.inference_mode():
        torch_out, torch_hidden = torch_gru(x, h0)
    np_out, np_hidden = custom.forward(x.numpy(), h0.squeeze(0).numpy())

    seq_err = np.max(np.abs(torch_out.numpy() - np_out))
    hid_err = np.max(np.abs(torch_hidden.squeeze(0).numpy() - np_hidden))

    print(f"Max abs diff (sequence): {seq_err:.6e}")
    print(f"Max abs diff (hidden):   {hid_err:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NumPy GRU vs torch.nn.GRU")
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison(args)
