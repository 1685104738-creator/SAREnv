"""Configuration types for physical polygon surface sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..common.grid import validate_projected_metre_crs

SURFACE_OVERLAP_LAST_DEFINED_WINS = "last_defined_wins"
SURFACE_PHOTON_KERNEL_MODEL_NAME = "planar_gamma_point_kernel_superposition"
SURFACE_ACTIVITY_DENSITY_UNIT = "Bq/m^2"
CELL_ACTIVITY_UNIT = "Bq"
PHOTON_KERNEL_UNIT = "photons/m^2/decay"
PHOTON_FLUENCE_RATE_UNIT = "photons/m^2/s"
COLLISION_AIR_KERMA_RATE_UNIT = "uGy/h"

# Fixed Cs-137 parameters for the current physical surface model. The nuclear
# data values below are from NNDC/ENSDF. HALF_LIFE_YEARS is metadata only: this
# static simulation does not evolve activity through radioactive decay.
RADIONUCLIDE_NAME = "Cs-137"
HALF_LIFE_YEARS = 30.007
GAMMA_ENERGY_MEV = 0.661657
GAMMA_YIELD_PER_DECAY = 0.851

# The surface ground-truth field is evaluated at 1 m above ground on the
# existing 1 m radiation raster. No second surface-resolution system exists.
OBSERVATION_HEIGHT_M = 1.0
RASTER_RESOLUTION_M = 1.0

# Homogeneous dry-air coefficients at 661.657 keV. The mass attenuation
# coefficient is derived from interpolated NIST dry-air data; multiplying it by
# AIR_DENSITY_KG_M3 gives the fixed linear coefficient MU_AIR_M_INV.
AIR_DENSITY_KG_M3 = 1.205
MASS_ATTENUATION_COEFF_AIR_M2_KG = 7.752572415e-3
MU_AIR_M_INV = 9.341849760075e-3
MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG = 2.931111765e-3

# Exact energy conversion and output-unit conversions.
MEV_TO_J = 1.602176634e-13
SECONDS_PER_HOUR = 3600.0
GY_TO_UGY = 1.0e6

# Standard Cs-137 surface-contamination scenario presets. Callers may still
# provide any explicit non-negative activity_density_bq_m2 to a polygon config.
LOW_ACTIVITY_DENSITY_BQ_M2 = 1.0e5
MEDIUM_ACTIVITY_DENSITY_BQ_M2 = 1.0e6
HIGH_ACTIVITY_DENSITY_BQ_M2 = 5.0e6

# Derived from the named physical constants above; the conversion function
# performs the expanded calculation rather than treating this as a second
# conversion implementation.
CS137_FLUENCE_TO_KERMA_UGY_H = (
    GAMMA_ENERGY_MEV
    * MEV_TO_J
    * MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG
    * SECONDS_PER_HOUR
    * GY_TO_UGY
)


def _validate_polygon(geometry: Polygon | MultiPolygon, name: str) -> None:
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError(f"{name} must be a Polygon or MultiPolygon.")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{name} must be non-empty and valid.")


def _validate_projected_crs(crs: str) -> None:
    validate_projected_metre_crs(crs)


def _validate_activity_density(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class UniformPolygonConfig:
    """One polygon with a uniform surface activity density in Bq/m^2."""

    source_id: str
    geometry: Polygon | MultiPolygon
    activity_density_bq_m2: float
    crs: str

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        _validate_polygon(self.geometry, "geometry")
        _validate_projected_crs(self.crs)
        _validate_activity_density(
            self.activity_density_bq_m2,
            "activity_density_bq_m2",
        )


@dataclass(frozen=True)
class SurfaceZone:
    """One ordered zone with a surface activity density in Bq/m^2."""

    zone_id: str
    geometry: Polygon | MultiPolygon
    activity_density_bq_m2: float

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("zone_id must be a non-empty string.")
        _validate_polygon(self.geometry, "zone geometry")
        _validate_activity_density(
            self.activity_density_bq_m2,
            "zone activity_density_bq_m2",
        )


@dataclass(frozen=True)
class ZonedPolygonConfig:
    """Ordered polygon zones using an explicit last-defined-wins rule."""

    source_id: str
    zones: tuple[SurfaceZone, ...]
    crs: str
    overlap_rule: str = SURFACE_OVERLAP_LAST_DEFINED_WINS

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        if not self.zones:
            raise ValueError("ZonedPolygonConfig requires at least one zone.")
        if not all(isinstance(zone, SurfaceZone) for zone in self.zones):
            raise TypeError("zones must contain SurfaceZone objects.")
        if len({zone.zone_id for zone in self.zones}) != len(self.zones):
            raise ValueError("Zone IDs must be unique.")
        _validate_projected_crs(self.crs)
        if self.overlap_rule != SURFACE_OVERLAP_LAST_DEFINED_WINS:
            raise ValueError("Only overlap_rule='last_defined_wins' is implemented.")


@dataclass(frozen=True)
class RadionuclideProfile:
    """Nuclear data required by a surface photon-emission profile."""

    name: str
    half_life_years: float
    gamma_energy_mev: float
    gamma_yield_per_decay: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Radionuclide name must be non-empty.")
        if not np.isfinite(self.half_life_years) or self.half_life_years <= 0:
            raise ValueError("Radionuclide half-life must be finite and positive.")
        if not np.isfinite(self.gamma_energy_mev) or self.gamma_energy_mev <= 0:
            raise ValueError("Gamma energy must be finite and positive.")
        if (
            not np.isfinite(self.gamma_yield_per_decay)
            or self.gamma_yield_per_decay < 0
        ):
            raise ValueError("Gamma yield must be finite and non-negative.")


CS137_PROFILE = RadionuclideProfile(
    name=RADIONUCLIDE_NAME,
    half_life_years=HALF_LIFE_YEARS,
    gamma_energy_mev=GAMMA_ENERGY_MEV,
    gamma_yield_per_decay=GAMMA_YIELD_PER_DECAY,
)


@dataclass(frozen=True)
class SurfacePhotonResponseConfig:
    """Fixed Cs-137 parameters for the planar primary-photon kernel."""

    radionuclide_profile: RadionuclideProfile = CS137_PROFILE
    observation_height_m: float = OBSERVATION_HEIGHT_M
    mu_air_m_inv: float = MU_AIR_M_INV

    def __post_init__(self) -> None:
        if not isinstance(self.radionuclide_profile, RadionuclideProfile):
            raise TypeError("radionuclide_profile must be a RadionuclideProfile.")
        if self.radionuclide_profile != CS137_PROFILE:
            raise ValueError("The current surface model supports only Cs-137.")
        if self.observation_height_m != OBSERVATION_HEIGHT_M:
            raise ValueError("The current observation height is fixed at 1.0 m.")
        if self.mu_air_m_inv != MU_AIR_M_INV:
            raise ValueError("The current Cs-137 air attenuation is fixed.")

    @property
    def gamma_yield_per_decay(self) -> float:
        """Return the Cs-137 main-line photon yield used by the kernel."""
        return self.radionuclide_profile.gamma_yield_per_decay

    @property
    def gamma_energy_mev(self) -> float:
        """Return the Cs-137 main-line photon energy in MeV."""
        return self.radionuclide_profile.gamma_energy_mev


# Backward import aliases only. The old core-radius/cutoff/normalisation
# constructor is intentionally not supported by the physical model.
BenchmarkSurfaceResponseConfig = SurfacePhotonResponseConfig
BENCHMARK_KERNEL_MODEL_NAME = SURFACE_PHOTON_KERNEL_MODEL_NAME


__all__ = [
    "BENCHMARK_KERNEL_MODEL_NAME",
    "BenchmarkSurfaceResponseConfig",
    "CELL_ACTIVITY_UNIT",
    "COLLISION_AIR_KERMA_RATE_UNIT",
    "CS137_FLUENCE_TO_KERMA_UGY_H",
    "CS137_PROFILE",
    "AIR_DENSITY_KG_M3",
    "GAMMA_ENERGY_MEV",
    "GAMMA_YIELD_PER_DECAY",
    "GY_TO_UGY",
    "HALF_LIFE_YEARS",
    "HIGH_ACTIVITY_DENSITY_BQ_M2",
    "LOW_ACTIVITY_DENSITY_BQ_M2",
    "MASS_ATTENUATION_COEFF_AIR_M2_KG",
    "MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG",
    "MEDIUM_ACTIVITY_DENSITY_BQ_M2",
    "MEV_TO_J",
    "MU_AIR_M_INV",
    "OBSERVATION_HEIGHT_M",
    "PHOTON_FLUENCE_RATE_UNIT",
    "PHOTON_KERNEL_UNIT",
    "RADIONUCLIDE_NAME",
    "RASTER_RESOLUTION_M",
    "RadionuclideProfile",
    "SECONDS_PER_HOUR",
    "SURFACE_ACTIVITY_DENSITY_UNIT",
    "SURFACE_OVERLAP_LAST_DEFINED_WINS",
    "SURFACE_PHOTON_KERNEL_MODEL_NAME",
    "SurfacePhotonResponseConfig",
    "SurfaceZone",
    "UniformPolygonConfig",
    "ZonedPolygonConfig",
]
