"""Prepare the VoiceBank-DEMAND dataset locally."""
from __future__ import annotations

import argparse
from pathlib import Path

from dataloader import DEFAULT_DATA_ROOT, prepare_voicebank_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache VoiceBank-DEMAND-16k")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Target directory for the prepared dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_voicebank_dataset(args.out)


if __name__ == "__main__":
    main()
