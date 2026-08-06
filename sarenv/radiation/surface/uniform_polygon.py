"""Uniform Polygon nominal surface source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import mapping

from ..common.grid import GridSpec
from ..common.metadata import NOMINAL_SURFACE_UNIT
from .config import UniformPolygonConfig
from .grid import rasterize_polygon_mask, validate_geometry_grid_crs


@dataclass(frozen=True)
class UniformPolygonSource:
    """Rasterize one deterministic uniform polygon nominal field."""

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
        """Return the 1 m nominal field; polygon exterior cells are zero."""
        validate_geometry_grid_crs(self.config.crs, grid)
        mask = rasterize_polygon_mask(self.config.geometry, grid)
        field = np.zeros(grid.shape, dtype=float)
        field[mask] = self.config.nominal_surface_field_uSv_h
        return field

    def source_metadata(self) -> dict[str, object]:
        """Return JSON-ready source metadata."""
        return {
            "source_id": self.config.source_id,
            "surface_type": self.surface_type,
            "crs": self.config.crs,
            "nominal_intensity_unit": NOMINAL_SURFACE_UNIT,
            "nominal_surface_field_uSv_h": (
                self.config.nominal_surface_field_uSv_h
            ),
            "geometry": mapping(self.config.geometry),
            "zone_definitions": [],
            "overlap_rule": "not_applicable",
        }


__all__ = ["UniformPolygonSource"]
