"""Configuration for analytic synthetic point radiation sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.grid import validate_projected_metre_crs


POINT_MODEL_NAME = "inverse_square_inspired_softened"
DEFAULT_POINT_REFERENCE_DISTANCE_M = 1.0
DEFAULT_POINT_CORE_RADIUS_M = 0.5


@dataclass(frozen=True)
class PointSourceConfig:
    """Deterministic parameters for one spatially compact radiation source."""

    source_id: str
    x_m: float
    y_m: float
    reference_excess_uSv_h: float
    crs: str
    source_height_m: float = 0.0
    reference_distance_m: float = DEFAULT_POINT_REFERENCE_DISTANCE_M
    core_radius_m: float = DEFAULT_POINT_CORE_RADIUS_M
    cutoff_radius_m: float | None = None
    unit: str = "uSv/h"
    model_name: str = POINT_MODEL_NAME

    def __post_init__(self) -> None:
        if not self.source_id or not isinstance(self.source_id, str):
            raise ValueError("source_id must be a non-empty string.")
        validate_projected_metre_crs(self.crs)
        numeric = {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "source_height_m": self.source_height_m,
            "reference_distance_m": self.reference_distance_m,
            "reference_excess_uSv_h": self.reference_excess_uSv_h,
            "core_radius_m": self.core_radius_m,
        }
        for name, value in numeric.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.reference_distance_m <= 0:
            raise ValueError("reference_distance_m must be greater than zero.")
        if self.reference_excess_uSv_h < 0:
            raise ValueError("reference_excess_uSv_h must be non-negative.")
        if self.core_radius_m <= 0:
            raise ValueError("core_radius_m must be greater than zero.")
        if self.cutoff_radius_m is not None and (
            not np.isfinite(self.cutoff_radius_m)
            or self.cutoff_radius_m <= 0
        ):
            raise ValueError("cutoff_radius_m must be positive when provided.")
        if self.model_name != POINT_MODEL_NAME:
            raise ValueError(f"Unsupported point model: {self.model_name}")
        if self.unit != "uSv/h":
            raise ValueError("Point-source dose-rate unit must be 'uSv/h'.")


__all__ = [
    "DEFAULT_POINT_CORE_RADIUS_M",
    "DEFAULT_POINT_REFERENCE_DISTANCE_M",
    "POINT_MODEL_NAME",
    "PointSourceConfig",
]
