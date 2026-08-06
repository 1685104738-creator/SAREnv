"""Configuration types for deterministic polygon surface sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..common.grid import validate_projected_metre_crs


SURFACE_OVERLAP_LAST_DEFINED_WINS = "last_defined_wins"
BENCHMARK_KERNEL_MODEL_NAME = "inverse_square_inspired_normalised"


def _validate_polygon(geometry: Polygon | MultiPolygon, name: str) -> None:
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError(f"{name} must be a Polygon or MultiPolygon.")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{name} must be non-empty and valid.")


def _validate_projected_crs(crs: str) -> None:
    validate_projected_metre_crs(crs)


@dataclass(frozen=True)
class UniformPolygonConfig:
    """One polygon with one nominal synthetic surface intensity."""

    source_id: str
    geometry: Polygon | MultiPolygon
    nominal_surface_field_uSv_h: float
    crs: str

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        _validate_polygon(self.geometry, "geometry")
        _validate_projected_crs(self.crs)
        if (
            not np.isfinite(self.nominal_surface_field_uSv_h)
            or self.nominal_surface_field_uSv_h < 0
        ):
            raise ValueError(
                "nominal_surface_field_uSv_h must be finite and non-negative."
            )


@dataclass(frozen=True)
class SurfaceZone:
    """One ordered zone within a Zoned Polygon source."""

    zone_id: str
    geometry: Polygon | MultiPolygon
    nominal_surface_field_uSv_h: float

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id must be a non-empty string.")
        _validate_polygon(self.geometry, "zone geometry")
        if (
            not np.isfinite(self.nominal_surface_field_uSv_h)
            or self.nominal_surface_field_uSv_h < 0
        ):
            raise ValueError(
                "zone nominal_surface_field_uSv_h must be finite and non-negative."
            )


@dataclass(frozen=True)
class ZonedPolygonConfig:
    """Ordered polygon zones using an explicit last-defined-wins rule."""

    source_id: str
    zones: tuple[SurfaceZone, ...]
    crs: str
    overlap_rule: str = SURFACE_OVERLAP_LAST_DEFINED_WINS

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        if not self.zones:
            raise ValueError("ZonedPolygonConfig requires at least one zone.")
        if not all(isinstance(zone, SurfaceZone) for zone in self.zones):
            raise TypeError("zones must contain SurfaceZone objects.")
        if len({zone.zone_id for zone in self.zones}) != len(self.zones):
            raise ValueError("Zone IDs must be unique.")
        _validate_projected_crs(self.crs)
        if self.overlap_rule != SURFACE_OVERLAP_LAST_DEFINED_WINS:
            raise ValueError(
                "Only overlap_rule='last_defined_wins' is implemented."
            )


@dataclass(frozen=True)
class BenchmarkSurfaceResponseConfig:
    """Explicit parameters for the non-physical benchmark response kernel."""

    core_radius_m: float
    cutoff_radius_m: float
    resolution_m: float = 1.0
    normalise_kernel: bool = True
    model_name: str = BENCHMARK_KERNEL_MODEL_NAME

    def __post_init__(self) -> None:
        if not np.isclose(self.resolution_m, 1.0):
            raise ValueError("Surface response resolution is fixed at 1.0 m.")
        if not np.isfinite(self.core_radius_m) or self.core_radius_m <= 0:
            raise ValueError("core_radius_m must be finite and greater than zero.")
        if not np.isfinite(self.cutoff_radius_m) or self.cutoff_radius_m <= 0:
            raise ValueError("cutoff_radius_m must be finite and greater than zero.")
        cell_radius = self.cutoff_radius_m / self.resolution_m
        if not np.isclose(cell_radius, round(cell_radius), atol=1e-9):
            raise ValueError(
                "cutoff_radius_m must be a whole number of 1 m cells so full "
                "convolution bounds match the configured cutoff exactly."
            )
        if self.normalise_kernel is not True:
            raise ValueError("The first version requires a normalised kernel.")
        if self.model_name != BENCHMARK_KERNEL_MODEL_NAME:
            raise ValueError(f"Unsupported surface kernel model: {self.model_name}")


__all__ = [
    "BENCHMARK_KERNEL_MODEL_NAME",
    "BenchmarkSurfaceResponseConfig",
    "SURFACE_OVERLAP_LAST_DEFINED_WINS",
    "SurfaceZone",
    "UniformPolygonConfig",
    "ZonedPolygonConfig",
]
