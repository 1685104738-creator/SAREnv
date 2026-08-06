"""Composition of independent analytic and raster radiation sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS

from ..common.grid import validate_projected_metre_crs
from ..point.source import PointSource
from ..surface.dose_patch import DoseRatePatch


@dataclass(frozen=True)
class CompositeRadiationField:
    """
    Query a deterministic total dose rate from independent source components.

    Background is owned here and is added exactly once. Point sources and
    surface patches contain excess dose rate only.
    """

    crs: str
    background_uSv_h: float = 0.20
    point_sources: tuple[PointSource, ...] = ()
    surface_patches: tuple[DoseRatePatch, ...] = ()

    def __post_init__(self) -> None:
        validate_projected_metre_crs(self.crs)
        if not np.isfinite(self.background_uSv_h) or self.background_uSv_h < 0:
            raise ValueError("background_uSv_h must be finite and non-negative.")
        if not all(isinstance(source, PointSource) for source in self.point_sources):
            raise TypeError("point_sources must contain PointSource objects.")
        if not all(
            isinstance(patch, DoseRatePatch) for patch in self.surface_patches
        ):
            raise TypeError("surface_patches must contain DoseRatePatch objects.")
        field_crs = CRS.from_user_input(self.crs)
        for source in self.point_sources:
            if not field_crs.equals(CRS.from_user_input(source.config.crs)):
                raise ValueError("Point-source CRS does not match composite CRS.")
        for patch in self.surface_patches:
            if not field_crs.equals(CRS.from_user_input(patch.grid.crs)):
                raise ValueError("Surface-patch CRS does not match composite CRS.")

    def query_excess_dose_rate(
        self,
        x_m: float,
        y_m: float,
        z_m: float | None = None,
        *,
        patch_method: str = "bilinear",
    ) -> float:
        """Return the sum of all source excess contributions in uSv/h."""
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            raise ValueError("Composite query coordinates must be finite.")
        excess = sum(
            source.query_excess_dose_rate(x_m, y_m, z_m)
            for source in self.point_sources
        )
        excess += sum(
            patch.query_excess_dose_rate(
                x_m, y_m, method=patch_method
            )
            for patch in self.surface_patches
        )
        return float(excess)

    def query_total_dose_rate(
        self,
        x_m: float,
        y_m: float,
        z_m: float | None = None,
        *,
        patch_method: str = "bilinear",
    ) -> float:
        """Return background plus every source contribution in uSv/h."""
        return float(
            self.background_uSv_h
            + self.query_excess_dose_rate(
                x_m,
                y_m,
                z_m,
                patch_method=patch_method,
            )
        )


__all__ = ["CompositeRadiationField"]
