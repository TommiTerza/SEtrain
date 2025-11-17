from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

RUN_DIR_PREFIX = "run_"
RUN_LOG_BASENAME = "raw"
RUN_CSV_NAME = "thresholds.csv"

ENCODER_BLOCKS = tuple(f"enctra{i}" for i in range(3))
DECODER_BLOCKS = tuple(f"dectra{i}" for i in range(3))
DP_BLOCKS = tuple(
    f"dp{dp}{stage}{idx}"
    for dp in (1, 2)
    for stage in ("intra", "inter")
    for idx in (1, 2)
)
COMPONENT_BASES = ENCODER_BLOCKS + DECODER_BLOCKS + DP_BLOCKS


def per_component_columns() -> List[str]:
    columns: List[str] = []
    for base in COMPONENT_BASES:
        columns.append(f"{base}_x")
        columns.append(f"{base}_h")
    return columns


def threshold_csv_columns(include_metric: bool = True) -> List[str]:
    columns = ["threshold", "global", "global_x", "global_h"]
    columns.extend(per_component_columns())
    if include_metric:
        columns.append("metric")
    return columns


def component_weight_map() -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for base in ENCODER_BLOCKS + DECODER_BLOCKS:
        weights[f"{base}_x"] = 8.0
        weights[f"{base}_h"] = 16.0
    for dp in (1, 2):
        for stage in ("intra", "inter"):
            for idx in (1, 2):
                prefix = f"dp{dp}{stage}{idx}"
                weights[f"{prefix}_x"] = 8.0
                weights[f"{prefix}_h"] = 8.0
    return weights


def parse_run_index(path: Path) -> Tuple[int, Path]:
    """Return (index, path) for sorting run directories like run_3."""
    name = path.name
    try:
        suffix = name.split("_", 1)[1]
        index = int(suffix)
    except (IndexError, ValueError):
        index = -1
    return index, path
