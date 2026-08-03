"""
Sweep signed fixed-point Qm.n formats for tanh and report approximation error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def build_qmn_formats(bit_widths: list[int], frac_bits_override: list[int] | None) -> list[dict[str, int | str]]:
    formats: list[dict[str, int | str]] = []
    for bits in bit_widths:
        if bits < 2:
            raise ValueError(f"bit width must be >= 2, got {bits}")
        n_values = list(range(bits)) if frac_bits_override is None else frac_bits_override
        for n in n_values:
            if n < 0 or n > bits - 1:
                raise ValueError(f"invalid frac bits n={n} for {bits}-bit signed format")
            m = bits - 1 - n
            formats.append({"bits": bits, "m": m, "n": n, "label": f"Q{m}.{n}"})
    return formats


def quantize_dequantize(values: np.ndarray, *, bits: int, frac_bits: int) -> tuple[np.ndarray, np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    scale = float(2.0 ** (-frac_bits))
    raw = np.round(values / scale)
    q = np.clip(raw, qmin, qmax)
    deq = q * scale
    clipped = (raw < qmin) | (raw > qmax)
    return deq.astype(np.float64, copy=False), clipped


def evaluate_tanh_formats(
    x_values: np.ndarray,
    formats: list[dict[str, int | str]],
    *,
    quantize_input: bool,
) -> list[dict[str, float | int | str]]:
    reference = np.tanh(x_values)
    rows: list[dict[str, float | int | str]] = []
    for fmt in formats:
        bits = int(fmt["bits"])
        frac_bits = int(fmt["n"])

        x_eval = x_values
        if quantize_input:
            x_eval, _ = quantize_dequantize(x_eval, bits=bits, frac_bits=frac_bits)

        y_float = np.tanh(x_eval)
        y_qdq, clipped = quantize_dequantize(y_float, bits=bits, frac_bits=frac_bits)

        error = y_qdq - reference
        mse = float(np.mean(error**2))
        rmse = float(math.sqrt(mse))
        mae = float(np.mean(np.abs(error)))
        max_abs = float(np.max(np.abs(error)))
        signal_power = float(np.mean(reference**2))
        snr_db = float(10.0 * math.log10((signal_power + 1e-20) / (mse + 1e-20)))
        sat_pct = float(100.0 * np.mean(clipped.astype(np.float64)))

        rows.append(
            {
                "q_format": str(fmt["label"]),
                "bits": bits,
                "m": int(fmt["m"]),
                "n": frac_bits,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "max_abs_error": max_abs,
                "snr_db": snr_db,
                "output_clip_pct": sat_pct,
            }
        )
    rows.sort(key=lambda row: (float(row["rmse"]), -int(row["n"])))
    return rows


def save_results(
    rows: list[dict[str, float | int | str]],
    *,
    json_path: Path | None,
    csv_path: Path | None,
    metadata: dict[str, object],
) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": metadata, "results": rows}
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON: {json_path}")

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "q_format",
            "bits",
            "m",
            "n",
            "mse",
            "rmse",
            "mae",
            "max_abs_error",
            "snr_db",
            "output_clip_pct",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Saved CSV:  {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Qm.n formats for tanh output quantization.")
    parser.add_argument(
        "--bit-widths",
        type=str,
        default="8",
        help='Comma-separated bit widths, e.g. "8" or "8,16".',
    )
    parser.add_argument(
        "--frac-bits",
        type=str,
        default="all",
        help='Fractional bits n to test. Use "all" or comma-separated values like "7,6,5".',
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=-8.0,
        help="Lower bound of sampled tanh input range.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=8.0,
        help="Upper bound of sampled tanh input range.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=200001,
        help="Number of uniformly sampled input points.",
    )
    parser.add_argument(
        "--quantize-input",
        action="store_true",
        help="Quantize/dequantize tanh input with the same Qm.n format before tanh.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="How many top formats to print (sorted by RMSE).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional output JSON file.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_points < 2:
        raise ValueError("--num-points must be >= 2")
    if not (args.x_min < args.x_max):
        raise ValueError("--x-min must be smaller than --x-max")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    bit_widths = parse_csv_ints(args.bit_widths)
    if not bit_widths:
        raise ValueError("No bit widths provided.")

    frac_flag = str(args.frac_bits).strip().lower()
    frac_bits_override = None if frac_flag in {"all", "none", ""} else parse_csv_ints(args.frac_bits)
    formats = build_qmn_formats(bit_widths, frac_bits_override)

    x_values = np.linspace(float(args.x_min), float(args.x_max), int(args.num_points), dtype=np.float64)
    rows = evaluate_tanh_formats(
        x_values,
        formats,
        quantize_input=bool(args.quantize_input),
    )

    print("Top tanh formats by RMSE:")
    for row in rows[: int(args.top_k)]:
        print(
            f"  {row['q_format']:>6} ({row['bits']}b): "
            f"rmse={row['rmse']:.8f}, mae={row['mae']:.8f}, "
            f"max_abs={row['max_abs_error']:.8f}, snr={row['snr_db']:.2f} dB, "
            f"clip={row['output_clip_pct']:.4f}%"
        )

    metadata = {
        "bit_widths": bit_widths,
        "frac_bits": "all" if frac_bits_override is None else frac_bits_override,
        "x_min": float(args.x_min),
        "x_max": float(args.x_max),
        "num_points": int(args.num_points),
        "quantize_input": bool(args.quantize_input),
    }
    save_results(
        rows,
        json_path=Path(args.json_out).expanduser() if args.json_out else None,
        csv_path=Path(args.csv_out).expanduser() if args.csv_out else None,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
