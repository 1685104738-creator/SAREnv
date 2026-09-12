"""Polygon-to-grid helpers for 1 m nominal surface fields."""

from __future__ import annotations

import numpy as np
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon

from ..common.grid import GridSpec


def validate_geometry_grid_crs(geometry_crs: str, grid: GridSpec) -> None:
    """Require the geometry and raster to use the same projected CRS."""
    if not CRS.from_user_input(geometry_crs).equals(CRS.from_user_input(grid.crs)):
        raise ValueError("Polygon source CRS does not match GridSpec CRS.")


def rasterize_polygon_mask(
    geometry: Polygon | MultiPolygon, grid: GridSpec
) -> np.ndarray:
    """
    Rasterize using cell-centre inclusion and return south-to-north row order.

    Rasterio creates north-up arrays, so the result is flipped vertically to
    satisfy GridSpec's explicit lower-origin contract.
    """
    transform = from_origin(
        grid.minx,
        grid.maxy,
        grid.resolution_m,
        grid.resolution_m,
    )
    north_up = rasterize(
        [(geometry, 1)],
        out_shape=grid.shape,
        fill=0,
        transform=transform,
        all_touched=False,
        dtype=np.uint8,
    )
    return np.flipud(north_up).astype(bool, copy=False)


__all__ = ["rasterize_polygon_mask", "validate_geometry_grid_crs"]
