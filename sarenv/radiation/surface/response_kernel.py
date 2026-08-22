"""Physical gamma photon-response kernel for planar surface sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.grid import GridSpec
from .config import (
    AIR_DENSITY_KG_M3,
    MASS_ATTENUATION_COEFF_AIR_M2_KG,
    PHOTON_KERNEL_UNIT,
    RASTER_RESOLUTION_M,
    SURFACE_PHOTON_KERNEL_MODEL_NAME,
    SurfacePhotonResponseConfig,
)


@dataclass(frozen=True)
class SurfacePhotonResponseKernel:
    """Per-Bq gamma response over all relative offsets in one source grid."""

    config: SurfacePhotonResponseConfig
    values: np.ndarray
    grid: GridSpec

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "values", array)

        if array.ndim != 2:
            raise ValueError("Surface photon-response kernel must be a 2D array.")
        if not isinstance(self.grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        source_height, source_width = self.grid.shape
        expected_shape = (2 * source_height - 1, 2 * source_width - 1)
        if array.shape != expected_shape:
            raise ValueError(
                "Kernel shape must cover every relative offset in the source grid."
            )
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(
                "Surface photon-response kernel must be finite and non-negative."
            )

    @classmethod
    def create(
        cls,
        config: SurfacePhotonResponseConfig,
        grid: GridSpec,
    ) -> "SurfacePhotonResponseKernel":
        """Build the full-domain per-Bq primary gamma photon kernel."""
        if not isinstance(config, SurfacePhotonResponseConfig):
            raise TypeError("config must be a SurfacePhotonResponseConfig.")
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        if grid.resolution_m != RASTER_RESOLUTION_M:
            raise ValueError("The Cs-137 surface raster resolution is fixed at 1.0 m.")

        row_offsets = np.arange(
            -(grid.height - 1),
            grid.height,
            dtype=np.float64,
        )
        column_offsets = np.arange(
            -(grid.width - 1),
            grid.width,
            dtype=np.float64,
        )
        dx_m, dy_m = np.meshgrid(
            column_offsets * grid.resolution_m,
            row_offsets * grid.resolution_m,
        )
        distance_m = np.sqrt(
            dx_m**2
            + dy_m**2
            + config.observation_height_m**2
        )
        values = (
            config.gamma_yield_per_decay
            * np.exp(-config.mu_air_m_inv * distance_m)
            / (4.0 * np.pi * distance_m**2)
        )
        return cls(
            config=config,
            values=np.asarray(values, dtype=np.float64),
            grid=grid,
        )

    @property
    def center_index(self) -> tuple[int, int]:
        """Return the zero-horizontal-offset row/column index."""
        return self.grid.height - 1, self.grid.width - 1

    def to_metadata(self) -> dict[str, object]:
        """Return the physical kernel definition and units."""
        return {
            "kernel_class": type(self).__name__,
            "kernel_model": SURFACE_PHOTON_KERNEL_MODEL_NAME,
            "kernel_formula": (
                "gamma_yield_per_decay * exp(-mu_air_m_inv * R) / "
                "(4*pi*R^2)"
            ),
            "distance_formula": "R=sqrt(dx^2+dy^2+observation_height_m^2)",
            "kernel_unit": PHOTON_KERNEL_UNIT,
            "radionuclide_name": self.config.radionuclide_profile.name,
            "half_life_years": self.config.radionuclide_profile.half_life_years,
            "half_life_usage": "metadata_only_static_simulation",
            "gamma_energy_mev": self.config.gamma_energy_mev,
            "observation_height_m": self.config.observation_height_m,
            "gamma_yield_per_decay": self.config.gamma_yield_per_decay,
            "air_density_kg_m3": AIR_DENSITY_KG_M3,
            "mass_attenuation_coeff_air_m2_kg": (
                MASS_ATTENUATION_COEFF_AIR_M2_KG
            ),
            "mu_air_m_inv": self.config.mu_air_m_inv,
            "resolution_m": self.grid.resolution_m,
            "kernel_shape": list(self.values.shape),
            "kernel_center_index": list(self.center_index),
            "kernel_support": "complete_relative_offsets_of_current_grid",
            "kernel_cutoff": None,
            "normalised": False,
            "physical_model": True,
        }


# Backward import alias only. The implementation is now physical and has no
# core radius, cutoff, smoothing, or normalisation behaviour.
BenchmarkSurfaceResponseKernel = SurfacePhotonResponseKernel


__all__ = [
    "BenchmarkSurfaceResponseKernel",
    "SurfacePhotonResponseKernel",
]
