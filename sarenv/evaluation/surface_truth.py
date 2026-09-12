"""Post-run height planes for the existing physical Cs-137 surface truth."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import fftconvolve

from ..radiation.surface.config import (
    GAMMA_YIELD_PER_DECAY,
    MU_AIR_M_INV,
    OBSERVATION_HEIGHT_M,
)
from ..radiation.surface.dose_patch import CollisionAirKermaRatePatch
from ..radiation.surface.kerma import (
    photon_fluence_to_collision_air_kerma_rate,
)
from ..radiation.surface.simulator import SurfaceSimulationResult


def _collision_air_kerma_plane(
    result: SurfaceSimulationResult,
    observation_height_m: float,
) -> tuple[CollisionAirKermaRatePatch, float]:
    """Evaluate the unchanged primary-gamma model at one explicit height."""
    if not math.isfinite(observation_height_m) or observation_height_m <= 0.0:
        raise ValueError("observation_height_m must be finite and positive.")
    grid = result.grid
    row_offsets = np.arange(-(grid.height - 1), grid.height, dtype=np.float64)
    column_offsets = np.arange(-(grid.width - 1), grid.width, dtype=np.float64)
    dx_m, dy_m = np.meshgrid(
        column_offsets * grid.resolution_m,
        row_offsets * grid.resolution_m,
    )
    distance_m = np.sqrt(dx_m**2 + dy_m**2 + observation_height_m**2)
    photon_kernel = (
        GAMMA_YIELD_PER_DECAY
        * np.exp(-MU_AIR_M_INV * distance_m)
        / (4.0 * np.pi * distance_m**2)
    )
    raw_fluence = np.asarray(
        fftconvolve(result.cell_activity_bq, photon_kernel, mode="same"),
        dtype=np.float64,
    )
    if not np.isfinite(raw_fluence).all():
        raise ValueError("Surface height-plane FFT produced non-finite values.")
    minimum_raw = float(raw_fluence.min())
    kerma = photon_fluence_to_collision_air_kerma_rate(
        np.maximum(raw_fluence, 0.0)
    )
    metadata = {
        **result.collision_air_kerma_metadata,
        "observation_height_m": float(observation_height_m),
        "evaluation_plane": True,
        "source_activity_reused_without_modification": True,
        "physical_formula_changed": False,
        "convolution_mode": "same",
        "minimum_raw_fft_value_before_clipping": minimum_raw,
    }
    return (
        CollisionAirKermaRatePatch(
            collision_air_kerma_rate_uGy_h=np.asarray(kerma, dtype=np.float64),
            grid=grid,
            metadata=metadata,
        ),
        minimum_raw,
    )


@dataclass(frozen=True)
class SurfaceKermaTruthField:
    """Query unit-safe Cs-137 collision-air-kerma truth at 1 m and UAV height.

    The 1 m plane is the unchanged production ``SurfaceSimulationResult``.
    The platform plane reuses the same activity raster, gamma yield,
    attenuation and kerma conversion; only the explicitly requested
    observation height in ``R=sqrt(dx^2+dy^2+height^2)`` changes.
    """

    result: SurfaceSimulationResult
    platform_altitude_m: float
    platform_patch: CollisionAirKermaRatePatch
    minimum_raw_platform_fft_value: float

    @classmethod
    def from_simulation(
        cls,
        result: SurfaceSimulationResult,
        *,
        platform_altitude_m: float,
    ) -> "SurfaceKermaTruthField":
        if not isinstance(result, SurfaceSimulationResult):
            raise TypeError("result must be a SurfaceSimulationResult.")
        patch, minimum_raw = _collision_air_kerma_plane(
            result,
            platform_altitude_m,
        )
        return cls(
            result=result,
            platform_altitude_m=float(platform_altitude_m),
            platform_patch=patch,
            minimum_raw_platform_fft_value=minimum_raw,
        )

    @property
    def ground_reference_height_m(self) -> float:
        return OBSERVATION_HEIGHT_M

    @property
    def quantity(self) -> str:
        return "collision_air_kerma_rate"

    @property
    def unit(self) -> str:
        return "uGy/h"

    def query_excess(self, x_m: float, y_m: float, z_m: float) -> float:
        """Query an explicit supported truth plane without unit conversion."""
        if math.isclose(z_m, self.ground_reference_height_m, abs_tol=1e-12):
            return self.result.collision_air_kerma_rate_patch.query_collision_air_kerma_rate(
                x_m,
                y_m,
            )
        if math.isclose(z_m, self.platform_altitude_m, abs_tol=1e-12):
            return self.platform_patch.query_collision_air_kerma_rate(x_m, y_m)
        raise ValueError(
            "Surface truth is available only at the saved 1 m plane and the "
            "explicit platform-altitude evaluation plane."
        )


__all__ = ["SurfaceKermaTruthField"]
