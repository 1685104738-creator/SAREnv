"""Independent metre-based grids for synthetic radiation data."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pyproj import CRS


RADIATION_RESOLUTION_M = 1.0
RASTER_ORIGIN = "lower"
RASTER_AXIS_ORDER = "row_y_column_x"


def validate_projected_metre_crs(crs: str) -> None:
    """Require a projected CRS whose horizontal axes use metres."""
    crs_object = CRS.from_user_input(crs)
    if not crs_object.is_projected:
        raise ValueError("Radiation grids require a projected CRS.")
    horizontal_axes = crs_object.axis_info[:2]
    if len(horizontal_axes) < 2 or not all(
        np.isclose(axis.unit_conversion_factor, 1.0)
        for axis in horizontal_axes
    ):
        raise ValueError("Radiation grid CRS horizontal units must be metres.")


@dataclass(frozen=True)
class GridSpec:
    """
    Spatial definition of a radiation raster.

    ``bounds`` are pixel-edge bounds ``(minx, miny, maxx, maxy)``. Row zero is
    the southern row and row indices increase with Northing. Column indices
    increase with Easting. The maximum x/y bounds are exclusive.
    """

    bounds: tuple[float, float, float, float]
    resolution_m: float
    crs: str
    width: int
    height: int
    origin: str = RASTER_ORIGIN
    axis_order: str = RASTER_AXIS_ORDER

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.bounds)
        object.__setattr__(self, "bounds", values)
        object.__setattr__(self, "resolution_m", float(self.resolution_m))
        object.__setattr__(self, "crs", str(self.crs))

        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError("Grid bounds must contain four finite values.")
        minx, miny, maxx, maxy = values
        if maxx <= minx or maxy <= miny:
            raise ValueError("Grid bounds must have positive width and height.")
        if not np.isclose(self.resolution_m, RADIATION_RESOLUTION_M):
            raise ValueError(
                "Radiation GridSpec resolution is fixed at 1.0 m in this version."
            )
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("Grid width must be an integer.")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise TypeError("Grid height must be an integer.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid width and height must be positive.")
        if not self.crs:
            raise ValueError("Grid CRS must be a non-empty string.")
        validate_projected_metre_crs(self.crs)
        if self.origin != RASTER_ORIGIN:
            raise ValueError("Radiation grids require origin='lower'.")
        if self.axis_order != RASTER_AXIS_ORDER:
            raise ValueError(
                "Radiation grids require axis_order='row_y_column_x'."
            )

        expected_width = (maxx - minx) / self.resolution_m
        expected_height = (maxy - miny) / self.resolution_m
        if not np.isclose(expected_width, self.width, atol=1e-8, rtol=0.0):
            raise ValueError("Grid width is inconsistent with bounds/resolution.")
        if not np.isclose(expected_height, self.height, atol=1e-8, rtol=0.0):
            raise ValueError("Grid height is inconsistent with bounds/resolution.")

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        crs: str,
        *,
        resolution_m: float = RADIATION_RESOLUTION_M,
        padding_m: float = 0.0,
    ) -> "GridSpec":
        """Create a 1 m grid by snapping edge bounds outwards to whole metres."""
        if not np.isfinite(padding_m) or padding_m < 0:
            raise ValueError("padding_m must be finite and non-negative.")
        if not np.isclose(resolution_m, RADIATION_RESOLUTION_M):
            raise ValueError("Radiation resolution is fixed at 1.0 m.")
        raw = tuple(float(value) for value in bounds)
        if len(raw) != 4 or not np.isfinite(raw).all():
            raise ValueError("bounds must contain four finite values.")
        minx, miny, maxx, maxy = raw
        if maxx <= minx or maxy <= miny:
            raise ValueError("bounds must have positive width and height.")

        minx = math.floor((minx - padding_m) / resolution_m) * resolution_m
        miny = math.floor((miny - padding_m) / resolution_m) * resolution_m
        maxx = math.ceil((maxx + padding_m) / resolution_m) * resolution_m
        maxy = math.ceil((maxy + padding_m) / resolution_m) * resolution_m
        width = int(round((maxx - minx) / resolution_m))
        height = int(round((maxy - miny) / resolution_m))
        return cls(
            bounds=(minx, miny, maxx, maxy),
            resolution_m=resolution_m,
            crs=crs,
            width=width,
            height=height,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return NumPy shape in row/column order."""
        return self.height, self.width

    @property
    def minx(self) -> float:
        return self.bounds[0]

    @property
    def miny(self) -> float:
        return self.bounds[1]

    @property
    def maxx(self) -> float:
        return self.bounds[2]

    @property
    def maxy(self) -> float:
        return self.bounds[3]

    def contains(self, x_m: float, y_m: float) -> bool:
        """Return whether a point is inside the half-open pixel-edge bounds."""
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            return False
        return self.minx <= x_m < self.maxx and self.miny <= y_m < self.maxy

    def row_col_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Return the projected coordinate of a cell centre."""
        if isinstance(row, bool) or isinstance(col, bool):
            raise TypeError("row and col must be integer indices.")
        if not isinstance(row, (int, np.integer)) or not isinstance(
            col, (int, np.integer)
        ):
            raise TypeError("row and col must be integer indices.")
        if not 0 <= row < self.height or not 0 <= col < self.width:
            raise IndexError("row/col lies outside the radiation grid.")
        x_m = self.minx + (int(col) + 0.5) * self.resolution_m
        y_m = self.miny + (int(row) + 0.5) * self.resolution_m
        return float(x_m), float(y_m)

    def world_to_row_col(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Map a projected coordinate to its containing cell."""
        if not self.contains(x_m, y_m):
            raise ValueError(
                "Coordinate lies outside the grid's half-open pixel-edge bounds."
            )
        col = int(math.floor((x_m - self.minx) / self.resolution_m))
        row = int(math.floor((y_m - self.miny) / self.resolution_m))
        return row, col

    def world_to_fractional_cell(
        self, x_m: float, y_m: float
    ) -> tuple[float, float]:
        """Return fractional row/column indices referenced to cell centres."""
        if not self.contains(x_m, y_m):
            raise ValueError("Coordinate lies outside radiation grid bounds.")
        col = (x_m - self.minx) / self.resolution_m - 0.5
        row = (y_m - self.miny) / self.resolution_m - 0.5
        return float(row), float(col)

    def cell_center_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return projected x/y coordinate arrays for every cell centre."""
        x = self.minx + (np.arange(self.width) + 0.5) * self.resolution_m
        y = self.miny + (np.arange(self.height) + 0.5) * self.resolution_m
        return np.meshgrid(x, y)

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-ready description of the grid contract."""
        return {
            "bounds": list(self.bounds),
            "resolution_m": self.resolution_m,
            "crs": self.crs,
            "width": self.width,
            "height": self.height,
            "shape": list(self.shape),
            "origin": self.origin,
            "axis_order": self.axis_order,
            "bounds_meaning": "pixel_edges",
            "maximum_bounds_inclusive": False,
            "row_direction": "increasing_northing",
            "column_direction": "increasing_easting",
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, object]) -> "GridSpec":
        """Restore a GridSpec from ``to_metadata`` output."""
        return cls(
            bounds=tuple(float(v) for v in metadata["bounds"]),
            resolution_m=float(metadata["resolution_m"]),
            crs=str(metadata["crs"]),
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            origin=str(metadata.get("origin", RASTER_ORIGIN)),
            axis_order=str(metadata.get("axis_order", RASTER_AXIS_ORDER)),
        )


__all__ = [
    "GridSpec",
    "RADIATION_RESOLUTION_M",
    "RASTER_AXIS_ORDER",
    "RASTER_ORIGIN",
    "validate_projected_metre_crs",
]
