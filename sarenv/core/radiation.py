"""Deprecated compatibility names for the former heatmap-aligned radiation API."""

from __future__ import annotations

from dataclasses import dataclass
import warnings


RADIATION_SOURCE_POINT = "point"
RADIATION_SOURCE_DISTRIBUTED = "distributed"
RADIATION_PLACEMENT_NEAR = "near"
RADIATION_PLACEMENT_MIDDLE = "middle"
RADIATION_PLACEMENT_FAR = "far"
RADIATION_INTENSITY_LOW = "low"
RADIATION_INTENSITY_MEDIUM = "medium"
RADIATION_INTENSITY_HIGH = "high"
RADIATION_SHAPE_CIRCULAR = "circular"
RADIATION_SHAPE_STRETCHED = "stretched"
DEFAULT_BACKGROUND_DOSE_RATE = 0.20

# Retained only so older imports fail with a migration message rather than an
# ImportError. These presets are not used by the new deterministic modules.
POINT_DOSE_RATE_PRESETS = {"low": 2_000.0, "medium": 20_000.0, "high": 200_000.0}
DISTRIBUTED_DOSE_RATE_PRESETS = {"low": 1.5, "medium": 5.0, "high": 12.8}
DISTRIBUTED_SHAPE_PRESETS = {
    "circular": {"sigma_x_m": 150.0, "sigma_y_m": 150.0},
    "stretched": {"sigma_x_m": 300.0, "sigma_y_m": 100.0},
}
PLACEMENT_DISTANCE_PRESETS = {
    "near": {"minimum": ("small", 0.25), "maximum": ("small", 0.75)},
    "middle": {"minimum": ("small", 1.0), "maximum": ("medium", 1.0)},
    "far": {"minimum": ("medium", 1.0), "maximum": ("large", 0.90)},
}


_MIGRATION_MESSAGE = (
    "The shared RadiationConfig/master-heatmap API is deprecated and disabled. "
    "Use sarenv.radiation.PointSourceConfig for analytic point sources, or "
    "UniformPolygonConfig/ZonedPolygonConfig with simulate_surface_source() "
    "for independent 1 m local surface patches."
)


@dataclass
class RadiationConfig:
    """Deprecated configuration retained only for import compatibility."""

    source_type: str = "point"
    placement_type: str = "near"
    intensity_type: str = "medium"
    shape_type: str = "circular"
    seed: int = 42
    background_dose_rate: float | None = None
    peak_dose_rate: float | None = None
    source_position: tuple[float, float] | None = None
    source_distance_m: float | None = None
    source_angle_deg: float | None = None
    sigma_x_m: float | None = None
    sigma_y_m: float | None = None
    source_count: int = 1
    reference_height_m: float = 1.0
    field_model: str = "default"
    environment_interaction: str = "none"
    time_model: str = "static"
    measurement_model: str = "ground_truth"

    def __post_init__(self) -> None:
        warnings.warn(_MIGRATION_MESSAGE, DeprecationWarning, stacklevel=2)


def generate_radiation_field(*args, **kwargs):
    """Reject the removed SAR-master-grid generator with migration guidance."""
    del args, kwargs
    warnings.warn(_MIGRATION_MESSAGE, DeprecationWarning, stacklevel=2)
    raise NotImplementedError(_MIGRATION_MESSAGE)


__all__ = [
    "DEFAULT_BACKGROUND_DOSE_RATE",
    "DISTRIBUTED_DOSE_RATE_PRESETS",
    "DISTRIBUTED_SHAPE_PRESETS",
    "PLACEMENT_DISTANCE_PRESETS",
    "POINT_DOSE_RATE_PRESETS",
    "RADIATION_INTENSITY_HIGH",
    "RADIATION_INTENSITY_LOW",
    "RADIATION_INTENSITY_MEDIUM",
    "RADIATION_PLACEMENT_FAR",
    "RADIATION_PLACEMENT_MIDDLE",
    "RADIATION_PLACEMENT_NEAR",
    "RADIATION_SHAPE_CIRCULAR",
    "RADIATION_SHAPE_STRETCHED",
    "RADIATION_SOURCE_DISTRIBUTED",
    "RADIATION_SOURCE_POINT",
    "RadiationConfig",
    "generate_radiation_field",
]
