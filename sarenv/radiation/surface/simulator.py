"""Deterministic surface activity rasterisation and FFT superposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS
from scipy.signal import fftconvolve

from ..common.grid import GridSpec
from ..common.metadata import RADIATION_GENERATOR_VERSION
from .config import (
    AIR_DENSITY_KG_M3,
    CELL_ACTIVITY_UNIT,
    COLLISION_AIR_KERMA_RATE_UNIT,
    CS137_FLUENCE_TO_KERMA_UGY_H,
    GY_TO_UGY,
    MASS_ATTENUATION_COEFF_AIR_M2_KG,
    MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG,
    MEV_TO_J,
    PHOTON_FLUENCE_RATE_UNIT,
    SECONDS_PER_HOUR,
    SURFACE_ACTIVITY_DENSITY_UNIT,
)
from .dose_patch import CollisionAirKermaRatePatch, PhotonFluenceRatePatch
from .kerma import photon_fluence_to_collision_air_kerma_rate
from .response_kernel import SurfacePhotonResponseKernel
from .uniform_polygon import UniformPolygonSource
from .zoned_polygon import ZonedPolygonSource

SurfaceSource = UniformPolygonSource | ZonedPolygonSource


@dataclass(frozen=True)
class SurfaceSimulationResult:
    """Physical surface-source inputs, kernel, output, and metadata."""

    activity_density_bq_m2: np.ndarray
    cell_activity_bq: np.ndarray
    grid: GridSpec
    kernel: SurfacePhotonResponseKernel
    photon_fluence_rate_patch: PhotonFluenceRatePatch
    collision_air_kerma_rate_patch: CollisionAirKermaRatePatch
    activity_metadata: dict[str, object]
    photon_fluence_metadata: dict[str, object]
    collision_air_kerma_metadata: dict[str, object]
    minimum_raw_fft_value_before_clipping: float

    def __post_init__(self) -> None:
        activity_density = np.asarray(
            self.activity_density_bq_m2,
            dtype=np.float64,
        )
        cell_activity = np.asarray(self.cell_activity_bq, dtype=np.float64)
        object.__setattr__(self, "activity_density_bq_m2", activity_density)
        object.__setattr__(self, "cell_activity_bq", cell_activity)
        if activity_density.shape != self.grid.shape:
            raise ValueError("Activity-density raster shape does not match its grid.")
        if cell_activity.shape != self.grid.shape:
            raise ValueError("Cell-activity raster shape does not match its grid.")
        if not np.isfinite(activity_density).all() or (activity_density < 0).any():
            raise ValueError("Activity-density raster must be finite and non-negative.")
        if not np.isfinite(cell_activity).all() or (cell_activity < 0).any():
            raise ValueError("Cell-activity raster must be finite and non-negative.")
        if not np.isfinite(self.minimum_raw_fft_value_before_clipping):
            raise ValueError("Raw FFT minimum must be finite.")


def simulate_surface_source(
    source: SurfaceSource,
    kernel: SurfacePhotonResponseKernel,
    *,
    nominal_grid: GridSpec | None = None,
) -> SurfaceSimulationResult:
    """
    Generate absolute photon fluence rate from a polygon surface source.

    The polygon raster stores surface activity density [Bq/m^2]. Multiplying
    it by cell area gives cell activity [Bq]. Linear FFT convolution with the
    per-Bq physical response kernel then gives photon fluence rate
    [photons/m^2/s] on the unchanged input grid.
    """
    if not isinstance(source, (UniformPolygonSource, ZonedPolygonSource)):
        raise TypeError("source must be UniformPolygonSource or ZonedPolygonSource.")
    if not isinstance(kernel, SurfacePhotonResponseKernel):
        raise TypeError("kernel must be a SurfacePhotonResponseKernel.")
    if nominal_grid is None:
        nominal_grid = GridSpec.from_bounds(source.bounds, source.crs)
    if not CRS.from_user_input(source.crs).equals(
        CRS.from_user_input(nominal_grid.crs)
    ):
        raise ValueError("Source and grid CRS do not match.")
    if nominal_grid.shape != kernel.grid.shape:
        raise ValueError("Grid shape does not match the kernel source-grid shape.")
    if nominal_grid.resolution_m != kernel.grid.resolution_m:
        raise ValueError("Grid and kernel resolutions do not match.")

    activity_density = np.asarray(
        source.rasterize(nominal_grid),
        dtype=np.float64,
    )
    if activity_density.shape != nominal_grid.shape:
        raise ValueError("Rasterized activity-density field shape is invalid.")
    if not np.isfinite(activity_density).all() or (activity_density < 0).any():
        raise ValueError("Activity density must be finite and non-negative.")

    cell_area_m2 = nominal_grid.resolution_m * nominal_grid.resolution_m
    cell_activity = np.asarray(
        activity_density * cell_area_m2,
        dtype=np.float64,
    )
    raw_photon_fluence_rate = np.asarray(
        fftconvolve(cell_activity, kernel.values, mode="same"),
        dtype=np.float64,
    )
    if not np.isfinite(raw_photon_fluence_rate).all():
        raise ValueError("FFT convolution produced NaN or infinite values.")
    minimum_raw_fft_value = float(raw_photon_fluence_rate.min())
    # Positive inputs have a non-negative physical result. FFT round-off can
    # produce tiny negative values, so only the sign is corrected; no scaling,
    # normalisation, threshold, or tunable numerical tolerance is applied.
    photon_fluence_rate = np.maximum(raw_photon_fluence_rate, 0.0)
    collision_air_kerma_rate = (
        photon_fluence_to_collision_air_kerma_rate(photon_fluence_rate)
    )

    source_metadata = source.source_metadata()
    activity_metadata = {
        "schema_version": 1,
        "model_type": "surface_activity_density_raster",
        "surface_type": source.surface_type,
        "quantity": "surface_activity_density",
        "activity_density_unit": SURFACE_ACTIVITY_DENSITY_UNIT,
        "cell_activity_unit": CELL_ACTIVITY_UNIT,
        "cell_area_m2": cell_area_m2,
        "total_activity_bq": float(cell_activity.sum()),
        "radionuclide_name": kernel.config.radionuclide_profile.name,
        "half_life_years": kernel.config.radionuclide_profile.half_life_years,
        "radioactive_decay_applied": False,
        "generator_version": RADIATION_GENERATOR_VERSION,
        "grid": nominal_grid.to_metadata(),
        "source": source_metadata,
    }
    photon_fluence_metadata = {
        "schema_version": 1,
        "model_type": "surface_gamma_point_kernel_superposition",
        "surface_type": source.surface_type,
        "quantity": "photon_fluence_rate",
        "output_unit": PHOTON_FLUENCE_RATE_UNIT,
        "background_included": False,
        "generator_version": RADIATION_GENERATOR_VERSION,
        "grid": nominal_grid.to_metadata(),
        "convolution_mode": "same",
        "convolution_type": "linear",
        "convolution_implementation": "scipy.signal.fftconvolve",
        "absolute_scaling_preserved": True,
        "normalised": False,
        "minimum_raw_fft_value_before_clipping": minimum_raw_fft_value,
        "cell_area_m2": cell_area_m2,
        "total_activity_bq": float(cell_activity.sum()),
        **kernel.to_metadata(),
        "source": source_metadata,
    }
    photon_patch = PhotonFluenceRatePatch(
        photon_fluence_rate=photon_fluence_rate,
        grid=nominal_grid,
        metadata=photon_fluence_metadata,
    )
    collision_air_kerma_metadata = {
        "schema_version": 1,
        "model_type": "cs137_collision_air_kerma_conversion",
        "surface_type": source.surface_type,
        "quantity": "collision_air_kerma_rate",
        "output_unit": COLLISION_AIR_KERMA_RATE_UNIT,
        "background_included": False,
        "generator_version": RADIATION_GENERATOR_VERSION,
        "grid": nominal_grid.to_metadata(),
        "radionuclide_name": kernel.config.radionuclide_profile.name,
        "half_life_years": kernel.config.radionuclide_profile.half_life_years,
        "half_life_usage": "metadata_only_static_simulation",
        "gamma_energy_mev": kernel.config.gamma_energy_mev,
        "gamma_yield_per_decay": kernel.config.gamma_yield_per_decay,
        "gamma_yield_applied_in_kerma_conversion": False,
        "gamma_yield_applied_stage": "activity_to_photon_fluence_kernel",
        "air_density_kg_m3": AIR_DENSITY_KG_M3,
        "mass_attenuation_coeff_air_m2_kg": MASS_ATTENUATION_COEFF_AIR_M2_KG,
        "mu_air_m_inv": kernel.config.mu_air_m_inv,
        "mass_energy_absorption_coeff_air_m2_kg": (
            MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG
        ),
        "mev_to_j": MEV_TO_J,
        "seconds_per_hour": SECONDS_PER_HOUR,
        "gy_to_ugy": GY_TO_UGY,
        "fluence_to_kerma_uGy_h_per_ph_m2_s": (
            CS137_FLUENCE_TO_KERMA_UGY_H
        ),
        "conversion_formula": (
            "fluence_rate * gamma_energy_mev * mev_to_j * "
            "mass_energy_absorption_coeff_air_m2_kg * seconds_per_hour * "
            "gy_to_ugy"
        ),
        "primary_photons_only": True,
        "dose_equivalent": False,
        "effective_dose": False,
    }
    kerma_patch = CollisionAirKermaRatePatch(
        collision_air_kerma_rate_uGy_h=collision_air_kerma_rate,
        grid=nominal_grid,
        metadata=collision_air_kerma_metadata,
    )
    return SurfaceSimulationResult(
        activity_density_bq_m2=activity_density,
        cell_activity_bq=cell_activity,
        grid=nominal_grid,
        kernel=kernel,
        photon_fluence_rate_patch=photon_patch,
        collision_air_kerma_rate_patch=kerma_patch,
        activity_metadata=activity_metadata,
        photon_fluence_metadata=photon_fluence_metadata,
        collision_air_kerma_metadata=collision_air_kerma_metadata,
        minimum_raw_fft_value_before_clipping=minimum_raw_fft_value,
    )


__all__ = ["SurfaceSimulationResult", "simulate_surface_source"]
