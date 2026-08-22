"""Ordered Zoned Polygon surface activity-density source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import mapping
from shapely.ops import unary_union

from ..common.grid import GridSpec
from .config import SURFACE_ACTIVITY_DENSITY_UNIT, ZonedPolygonConfig
from .grid import rasterize_polygon_mask, validate_geometry_grid_crs


@dataclass(frozen=True)
class ZonedPolygonSource:
    """Rasterize ordered zones; later zones explicitly replace earlier zones."""

    config: ZonedPolygonConfig

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        geometry = unary_union([zone.geometry for zone in self.config.zones])
        return tuple(float(value) for value in geometry.bounds)

    @property
    def crs(self) -> str:
        return self.config.crs

    @property
    def surface_type(self) -> str:
        return "zoned_polygon"

    def rasterize(self, grid: GridSpec) -> np.ndarray:
        """Return activity density [Bq/m^2] using last-defined-zone-wins."""
        validate_geometry_grid_crs(self.config.crs, grid)
        field = np.zeros(grid.shape, dtype=np.float64)
        for zone in self.config.zones:
            mask = rasterize_polygon_mask(zone.geometry, grid)
            field[mask] = zone.activity_density_bq_m2
        return field

    def source_metadata(self) -> dict[str, object]:
        """Return ordered zone definitions and the explicit overlap rule."""
        zones = [
            {
                "order": index,
                "zone_id": zone.zone_id,
                "activity_density_bq_m2": zone.activity_density_bq_m2,
                "geometry": mapping(zone.geometry),
            }
            for index, zone in enumerate(self.config.zones)
        ]
        return {
            "source_id": self.config.source_id,
            "surface_type": self.surface_type,
            "crs": self.config.crs,
            "quantity": "surface_activity_density",
            "activity_density_unit": SURFACE_ACTIVITY_DENSITY_UNIT,
            "zone_definitions": zones,
            "overlap_rule": self.config.overlap_rule,
        }


__all__ = ["ZonedPolygonSource"]
