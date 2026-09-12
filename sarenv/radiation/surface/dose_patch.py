"""Local surface-radiation raster query interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..common.grid import GridSpec


@dataclass(frozen=True)
class DoseRatePatch:
    """A local 1 m excess-only radiation patch with bilinear queries."""

    excess_uSv_h: np.ndarray
    grid: GridSpec
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.excess_uSv_h, dtype=float)
        object.__setattr__(self, "excess_uSv_h", array)
        if array.shape != self.grid.shape:
            message = (
                f"Patch array shape {array.shape} does not match "
                f"grid {self.grid.shape}."
            )
            raise ValueError(
                message
            )
        if not np.isfinite(array).all():
            raise ValueError("Dose-rate patch contains NaN or infinite values.")
        if (array < 0).any():
            raise ValueError("Dose-rate patch excess values must be non-negative.")
        if self.metadata.get("background_included", False):
            raise ValueError("DoseRatePatch must contain source excess only.")

    def query_excess_dose_rate(
        self, x_m: float, y_m: float, *, method: str = "bilinear"
    ) -> float:
        """
        Query excess at projected coordinates.

        Outside the half-open patch bounds, including an exact maximum x/y
        boundary, the result is zero. Coordinates between an edge and the
        nearest cell centre use the nearest edge cell during interpolation.
        """
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            raise ValueError("Patch query coordinates must be finite.")
        if not self.grid.contains(float(x_m), float(y_m)):
            return 0.0
        if method == "nearest":
            row, col = self.grid.world_to_row_col(float(x_m), float(y_m))
            return float(self.excess_uSv_h[row, col])
        if method != "bilinear":
            raise ValueError("method must be 'bilinear' or 'nearest'.")

        row_f, col_f = self.grid.world_to_fractional_cell(
            float(x_m), float(y_m)
        )
        row_f = float(np.clip(row_f, 0.0, self.grid.height - 1.0))
        col_f = float(np.clip(col_f, 0.0, self.grid.width - 1.0))
        row0 = int(math.floor(row_f))
        col0 = int(math.floor(col_f))
        row1 = min(row0 + 1, self.grid.height - 1)
        col1 = min(col0 + 1, self.grid.width - 1)
        row_weight = row_f - row0
        col_weight = col_f - col0

        lower = (
            (1.0 - col_weight) * self.excess_uSv_h[row0, col0]
            + col_weight * self.excess_uSv_h[row0, col1]
        )
        upper = (
            (1.0 - col_weight) * self.excess_uSv_h[row1, col0]
            + col_weight * self.excess_uSv_h[row1, col1]
        )
        return float((1.0 - row_weight) * lower + row_weight * upper)


@dataclass(frozen=True)
class PhotonFluenceRatePatch:
    """A photon fluence-rate raster in photons/m^2/s with spatial queries."""

    photon_fluence_rate: np.ndarray
    grid: GridSpec
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.photon_fluence_rate, dtype=np.float64)
        object.__setattr__(self, "photon_fluence_rate", array)
        if array.shape != self.grid.shape:
            message = (
                f"Patch array shape {array.shape} does not match "
                f"grid {self.grid.shape}."
            )
            raise ValueError(message)
        if not np.isfinite(array).all():
            raise ValueError(
                "Photon fluence-rate patch contains NaN or infinite values."
            )
        if (array < 0).any():
            raise ValueError("Photon fluence-rate values must be non-negative.")

    def query_photon_fluence_rate(
        self, x_m: float, y_m: float, *, method: str = "bilinear"
    ) -> float:
        """Query photon fluence rate at projected coordinates."""
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            raise ValueError("Patch query coordinates must be finite.")
        if not self.grid.contains(float(x_m), float(y_m)):
            return 0.0
        if method == "nearest":
            row, col = self.grid.world_to_row_col(float(x_m), float(y_m))
            return float(self.photon_fluence_rate[row, col])
        if method != "bilinear":
            raise ValueError("method must be 'bilinear' or 'nearest'.")

        row_f, col_f = self.grid.world_to_fractional_cell(float(x_m), float(y_m))
        row_f = float(np.clip(row_f, 0.0, self.grid.height - 1.0))
        col_f = float(np.clip(col_f, 0.0, self.grid.width - 1.0))
        row0 = int(math.floor(row_f))
        col0 = int(math.floor(col_f))
        row1 = min(row0 + 1, self.grid.height - 1)
        col1 = min(col0 + 1, self.grid.width - 1)
        row_weight = row_f - row0
        col_weight = col_f - col0

        lower = (
            (1.0 - col_weight) * self.photon_fluence_rate[row0, col0]
            + col_weight * self.photon_fluence_rate[row0, col1]
        )
        upper = (
            (1.0 - col_weight) * self.photon_fluence_rate[row1, col0]
            + col_weight * self.photon_fluence_rate[row1, col1]
        )
        return float((1.0 - row_weight) * lower + row_weight * upper)


@dataclass(frozen=True)
class CollisionAirKermaRatePatch:
    """A collision air kerma-rate raster in uGy/h with spatial queries."""

    collision_air_kerma_rate_uGy_h: np.ndarray
    grid: GridSpec
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(
            self.collision_air_kerma_rate_uGy_h,
            dtype=np.float64,
        )
        object.__setattr__(self, "collision_air_kerma_rate_uGy_h", array)
        if array.shape != self.grid.shape:
            message = (
                f"Patch array shape {array.shape} does not match "
                f"grid {self.grid.shape}."
            )
            raise ValueError(message)
        if not np.isfinite(array).all():
            raise ValueError(
                "Collision air kerma-rate patch contains NaN or infinite values."
            )
        if (array < 0.0).any():
            raise ValueError(
                "Collision air kerma-rate values must be non-negative."
            )

    def query_collision_air_kerma_rate(
        self, x_m: float, y_m: float, *, method: str = "bilinear"
    ) -> float:
        """Query collision air kerma rate at projected coordinates."""
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            raise ValueError("Patch query coordinates must be finite.")
        if not self.grid.contains(float(x_m), float(y_m)):
            return 0.0
        if method == "nearest":
            row, col = self.grid.world_to_row_col(float(x_m), float(y_m))
            return float(self.collision_air_kerma_rate_uGy_h[row, col])
        if method != "bilinear":
            raise ValueError("method must be 'bilinear' or 'nearest'.")

        row_f, col_f = self.grid.world_to_fractional_cell(float(x_m), float(y_m))
        row_f = float(np.clip(row_f, 0.0, self.grid.height - 1.0))
        col_f = float(np.clip(col_f, 0.0, self.grid.width - 1.0))
        row0 = int(math.floor(row_f))
        col0 = int(math.floor(col_f))
        row1 = min(row0 + 1, self.grid.height - 1)
        col1 = min(col0 + 1, self.grid.width - 1)
        row_weight = row_f - row0
        col_weight = col_f - col0

        lower = (
            (1.0 - col_weight)
            * self.collision_air_kerma_rate_uGy_h[row0, col0]
            + col_weight * self.collision_air_kerma_rate_uGy_h[row0, col1]
        )
        upper = (
            (1.0 - col_weight)
            * self.collision_air_kerma_rate_uGy_h[row1, col0]
            + col_weight * self.collision_air_kerma_rate_uGy_h[row1, col1]
        )
        return float((1.0 - row_weight) * lower + row_weight * upper)


__all__ = [
    "CollisionAirKermaRatePatch",
    "DoseRatePatch",
    "PhotonFluenceRatePatch",
]
