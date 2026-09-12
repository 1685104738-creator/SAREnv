"""JSON metadata helpers shared by radiation modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RADIATION_GENERATOR_VERSION = "2.0"
DOSE_RATE_UNIT = "uSv/h"
EXCESS_DOSE_RATE_QUANTITY = "synthetic_excess_gamma_dose_rate"
NOMINAL_SURFACE_UNIT = "synthetic_nominal_surface_field_uSv_h"


def to_json_ready(value: Any) -> Any:
    """Recursively convert NumPy values and tuples to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_metadata(path: str | Path, metadata: dict[str, object]) -> Path:
    """Write deterministic, sorted JSON metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(to_json_ready(metadata), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return output_path


def read_metadata(path: str | Path) -> dict[str, object]:
    """Read and validate one metadata JSON object."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {input_path}.")
    return value


__all__ = [
    "DOSE_RATE_UNIT",
    "EXCESS_DOSE_RATE_QUANTITY",
    "NOMINAL_SURFACE_UNIT",
    "RADIATION_GENERATOR_VERSION",
    "read_metadata",
    "to_json_ready",
    "write_metadata",
]
