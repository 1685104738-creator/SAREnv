"""Uniform Polygon surface activity-density source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import mapping

from ..common.grid import GridSpec
from .config import SURFACE_ACTIVITY_DENSITY_UNIT, UniformPolygonConfig
from .grid import rasterize_polygon_mask, validate_geometry_grid_crs


@dataclass(frozen=True)
class UniformPolygonSource:
    """Rasterize one deterministic uniform activity-density polygon."""

    config: UniformPolygonConfig

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in self.config.geometry.bounds)

    @property
    def crs(self) -> str:
        return self.config.crs

    @property
    def surface_type(self) -> str:
        return "uniform_polygon"

    def rasterize(self, grid: GridSpec) -> np.ndarray:
        """Return activity density [Bq/m^2]; polygon exterior cells are zero."""
        validate_geometry_grid_crs(self.config.crs, grid)
        mask = rasterize_polygon_mask(self.config.geometry, grid)
        field = np.zeros(grid.shape, dtype=np.float64)
        field[mask] = self.config.activity_density_bq_m2
        return field

    def source_metadata(self) -> dict[str, object]:
        """Return JSON-ready source metadata."""
        return {
            "source_id": self.config.source_id,
            "surface_type": self.surface_type,
            "crs": self.config.crs,
            "quantity": "surface_activity_density",
            "activity_density_unit": SURFACE_ACTIVITY_DENSITY_UNIT,
            "activity_density_bq_m2": self.config.activity_density_bq_m2,
            "geometry": mapping(self.config.geometry),
            "zone_definitions": [],
            "overlap_rule": "not_applicable",
        }


__all__ = ["UniformPolygonSource"]
