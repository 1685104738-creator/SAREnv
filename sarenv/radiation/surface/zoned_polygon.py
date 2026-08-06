"""Ordered Zoned Polygon nominal surface source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import mapping
from shapely.ops import unary_union

from ..common.grid import GridSpec
from ..common.metadata import NOMINAL_SURFACE_UNIT
from .config import ZonedPolygonConfig
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
        """Return a 1 m nominal field using last-defined-zone-wins overlap."""
        validate_geometry_grid_crs(self.config.crs, grid)
        field = np.zeros(grid.shape, dtype=float)
        for zone in self.config.zones:
            mask = rasterize_polygon_mask(zone.geometry, grid)
            field[mask] = zone.nominal_surface_field_uSv_h
        return field

    def source_metadata(self) -> dict[str, object]:
        """Return ordered zone definitions and the explicit overlap rule."""
        zones = [
            {
                "order": index,
                "zone_id": zone.zone_id,
                "nominal_surface_field_uSv_h": (
                    zone.nominal_surface_field_uSv_h
                ),
                "geometry": mapping(zone.geometry),
            }
            for index, zone in enumerate(self.config.zones)
        ]
        return {
            "source_id": self.config.source_id,
            "surface_type": self.surface_type,
            "crs": self.config.crs,
            "nominal_intensity_unit": NOMINAL_SURFACE_UNIT,
            "zone_definitions": zones,
            "overlap_rule": self.config.overlap_rule,
        }


__all__ = ["ZonedPolygonSource"]
