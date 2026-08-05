"""Synthetic radiation ground-truth generation for SAREnv."""

# Clear validation messages are kept inline to match the surrounding SAREnv style.
# ruff: noqa: EM101, EM102

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

from ..utils.lost_person_behavior import get_environment_radius_by_size

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

RADIATION_FIELD_MODEL_DEFAULT = "default"
RADIATION_ENVIRONMENT_INTERACTION_NONE = "none"
RADIATION_TIME_MODEL_STATIC = "static"
RADIATION_MEASUREMENT_MODEL_GROUND_TRUTH = "ground_truth"

GROUND_REFERENCE_QUANTITY = "ground_reference_gamma_dose_rate"
DOSE_RATE_UNIT = "uSv/h"

# Adjustable synthetic benchmark background; not a fixed UK background value or
# a nuclear-safety standard.
DEFAULT_BACKGROUND_DOSE_RATE = 0.20

# These dose-rate values are literature-anchored, adjustable synthetic benchmark
# defaults. They are not medical, regulatory, or official hazard classifications.
POINT_DOSE_RATE_PRESETS = {
    RADIATION_INTENSITY_LOW: 2_000.0,  # 2 mSv/h
    RADIATION_INTENSITY_MEDIUM: 20_000.0,  # 20 mSv/h
    RADIATION_INTENSITY_HIGH: 200_000.0,  # 200 mSv/h
}

# Literature-anchored, provisional lower dose-rate magnitudes for extended-area
# contamination benchmarks; these are not official safety classifications.
DISTRIBUTED_DOSE_RATE_PRESETS = {
    RADIATION_INTENSITY_LOW: 1.5,
    RADIATION_INTENSITY_MEDIUM: 5.0,
    RADIATION_INTENSITY_HIGH: 12.8,
}

# These fixed-metre geometries are provisional synthetic benchmark defaults,
# rather than universal contamination dimensions.
DISTRIBUTED_SHAPE_PRESETS = {
    RADIATION_SHAPE_CIRCULAR: {
        "sigma_x_m": 150.0,
        "sigma_y_m": 150.0,
    },
    RADIATION_SHAPE_STRETCHED: {
        "sigma_x_m": 300.0,
        "sigma_y_m": 100.0,
    },
}

# Each endpoint is (SAREnv size name, multiplier). Placement describes source
# distance from the IPP; it is intentionally independent of source field size.
PLACEMENT_DISTANCE_PRESETS = {
    RADIATION_PLACEMENT_NEAR: {
        "minimum": ("small", 0.25),
        "maximum": ("small", 0.75),
    },
    RADIATION_PLACEMENT_MIDDLE: {
        "minimum": ("small", 1.0),
        "maximum": ("medium", 1.0),
    },
    RADIATION_PLACEMENT_FAR: {
        "minimum": ("medium", 1.0),
        "maximum": ("large", 0.90),
    },
}

POINT_FIELD_MODEL_NAME = "inverse_square_like"
DISTRIBUTED_FIELD_MODEL_NAME = "gaussian"

_SUPPORTED_SOURCE_TYPES = {
    RADIATION_SOURCE_POINT,
    RADIATION_SOURCE_DISTRIBUTED,
}
_SUPPORTED_PLACEMENT_TYPES = {
    RADIATION_PLACEMENT_NEAR,
    RADIATION_PLACEMENT_MIDDLE,
    RADIATION_PLACEMENT_FAR,
}
_SUPPORTED_INTENSITY_TYPES = {
    RADIATION_INTENSITY_LOW,
    RADIATION_INTENSITY_MEDIUM,
    RADIATION_INTENSITY_HIGH,
}
_SUPPORTED_SHAPE_TYPES = {
    RADIATION_SHAPE_CIRCULAR,
    RADIATION_SHAPE_STRETCHED,
}


@dataclass
class RadiationConfig:
    """Configuration for one static synthetic radiation ground-truth scenario."""

    source_type: str = RADIATION_SOURCE_POINT
    placement_type: str = RADIATION_PLACEMENT_NEAR
    intensity_type: str = RADIATION_INTENSITY_MEDIUM
    shape_type: str = RADIATION_SHAPE_CIRCULAR

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

    field_model: str = RADIATION_FIELD_MODEL_DEFAULT
    environment_interaction: str = RADIATION_ENVIRONMENT_INTERACTION_NONE
    time_model: str = RADIATION_TIME_MODEL_STATIC
    measurement_model: str = RADIATION_MEASUREMENT_MODEL_GROUND_TRUTH

    def validate(self) -> None:
        """Validate supported MVP options and explicit override values."""
        if self.source_type not in _SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                f"Unsupported source_type '{self.source_type}'. "
                f"Expected one of {sorted(_SUPPORTED_SOURCE_TYPES)}."
            )
        if self.placement_type not in _SUPPORTED_PLACEMENT_TYPES:
            raise ValueError(
                f"Unsupported placement_type '{self.placement_type}'. "
                f"Expected one of {sorted(_SUPPORTED_PLACEMENT_TYPES)}."
            )
        if self.intensity_type not in _SUPPORTED_INTENSITY_TYPES:
            raise ValueError(
                f"Unsupported intensity_type '{self.intensity_type}'. "
                f"Expected one of {sorted(_SUPPORTED_INTENSITY_TYPES)}."
            )
        if self.shape_type not in _SUPPORTED_SHAPE_TYPES:
            raise ValueError(
                f"Unsupported shape_type '{self.shape_type}'. "
                f"Expected one of {sorted(_SUPPORTED_SHAPE_TYPES)}."
            )
        if self.source_count != 1:
            raise NotImplementedError(
                "Synthetic radiation MVP supports exactly one source; "
                f"received source_count={self.source_count}."
            )
        if not isinstance(self.seed, int | np.integer) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")

        _validate_optional_finite_nonnegative(
            self.background_dose_rate, "background_dose_rate"
        )
        _validate_optional_finite(self.peak_dose_rate, "peak_dose_rate")
        _validate_optional_finite_nonnegative(
            self.source_distance_m, "source_distance_m"
        )
        _validate_optional_positive(self.sigma_x_m, "sigma_x_m")
        _validate_optional_positive(self.sigma_y_m, "sigma_y_m")

        if not np.isfinite(self.reference_height_m) or self.reference_height_m <= 0:
            raise ValueError("reference_height_m must be a finite value greater than 0.")

        distance_provided = self.source_distance_m is not None
        angle_provided = self.source_angle_deg is not None
        if distance_provided != angle_provided:
            raise ValueError(
                "source_distance_m and source_angle_deg must be provided together."
            )
        if angle_provided and not np.isfinite(self.source_angle_deg):
            raise ValueError("source_angle_deg must be finite.")

        if self.source_position is not None:
            if len(self.source_position) != 2:
                raise ValueError(
                    "source_position must be a (longitude, latitude) WGS84 pair."
                )
            longitude, latitude = self.source_position
            if not np.isfinite(longitude) or not np.isfinite(latitude):
                raise ValueError("source_position coordinates must be finite.")
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(
                    "source_position must contain valid WGS84 longitude/latitude."
                )

        if self.field_model != RADIATION_FIELD_MODEL_DEFAULT:
            raise NotImplementedError(
                "Only field_model='default' is implemented in the radiation MVP."
            )
        if (
            self.environment_interaction
            != RADIATION_ENVIRONMENT_INTERACTION_NONE
        ):
            raise NotImplementedError(
                "Only environment_interaction='none' is implemented in the "
                "radiation MVP."
            )
        if self.time_model != RADIATION_TIME_MODEL_STATIC:
            raise NotImplementedError(
                "Only time_model='static' is implemented in the radiation MVP."
            )
        if (
            self.measurement_model
            != RADIATION_MEASUREMENT_MODEL_GROUND_TRUTH
        ):
            raise NotImplementedError(
                "Only measurement_model='ground_truth' is implemented in the "
                "radiation MVP; UAV (x, y, z) measurement is not implemented."
            )


@dataclass(frozen=True)
class ResolvedSourcePosition:
    """One source position resolved in both WGS84 and the master projected CRS."""

    position_wgs84: tuple[float, float]
    position_projected: tuple[float, float]
    distance_m: float
    angle_deg: float


@dataclass(frozen=True)
class ResolvedRadiationSource:
    """Resolved parameters for one synthetic radiation source."""

    source_type: str
    position: ResolvedSourcePosition
    peak_dose_rate: float
    sigma_x_m: float | None = None
    sigma_y_m: float | None = None


@dataclass(frozen=True)
class RadiationGenerationResult:
    """Generated master radiation raster, metadata, and resolved source list."""

    field: np.ndarray
    metadata: dict[str, object]
    sources: tuple[ResolvedRadiationSource, ...]


def _validate_optional_finite(value: float | None, name: str) -> None:
    if value is not None and not np.isfinite(value):
        raise ValueError(f"{name} must be finite when provided.")


def _validate_optional_finite_nonnegative(
    value: float | None, name: str
) -> None:
    if value is not None and (not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and greater than or equal to 0.")


def _validate_optional_positive(value: float | None, name: str) -> None:
    if value is not None and (not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and greater than 0.")


def resolve_background_dose_rate(explicit_background: float | None) -> float:
    """Resolve and validate the scenario background dose rate in uSv/h."""
    background = (
        DEFAULT_BACKGROUND_DOSE_RATE
        if explicit_background is None
        else float(explicit_background)
    )
    if not np.isfinite(background) or background < 0:
        raise ValueError(
            "Resolved background_dose_rate must be finite and non-negative."
        )
    return background


def resolve_peak_dose_rate(
    source_type: str,
    intensity_type: str,
    explicit_peak: float | None = None,
    background_dose_rate: float = DEFAULT_BACKGROUND_DOSE_RATE,
) -> float:
    """Resolve a source-type-specific benchmark peak dose rate in uSv/h."""
    if source_type == RADIATION_SOURCE_POINT:
        presets = POINT_DOSE_RATE_PRESETS
    elif source_type == RADIATION_SOURCE_DISTRIBUTED:
        presets = DISTRIBUTED_DOSE_RATE_PRESETS
    else:
        raise ValueError(f"Unsupported source_type '{source_type}'.")

    if intensity_type not in presets:
        raise ValueError(
            f"Unsupported intensity_type '{intensity_type}' for source_type "
            f"'{source_type}'."
        )

    peak = presets[intensity_type] if explicit_peak is None else float(explicit_peak)
    if not np.isfinite(peak) or peak <= background_dose_rate:
        raise ValueError(
            "Resolved peak_dose_rate must be finite and greater than "
            "background_dose_rate."
        )
    return peak


def resolve_placement_distance_range(
    placement_type: str,
    environment_type: str,
    environment_climate: str,
) -> tuple[float, float]:
    """Resolve a placement preset to a distance interval in metres."""
    if placement_type not in PLACEMENT_DISTANCE_PRESETS:
        raise ValueError(f"Unsupported placement_type '{placement_type}'.")

    rule = PLACEMENT_DISTANCE_PRESETS[placement_type]
    minimum_size, minimum_multiplier = rule["minimum"]
    maximum_size, maximum_multiplier = rule["maximum"]
    minimum_m = (
        get_environment_radius_by_size(
            environment_type, environment_climate, minimum_size
        )
        * 1000.0
        * minimum_multiplier
    )
    maximum_m = (
        get_environment_radius_by_size(
            environment_type, environment_climate, maximum_size
        )
        * 1000.0
        * maximum_multiplier
    )
    if (
        not np.isfinite(minimum_m)
        or not np.isfinite(maximum_m)
        or minimum_m < 0
        or maximum_m < minimum_m
    ):
        raise ValueError(
            f"Invalid resolved placement interval ({minimum_m}, {maximum_m}) m."
        )
    return float(minimum_m), float(maximum_m)


def resolve_source_position(
    config: RadiationConfig,
    center_point: tuple[float, float],
    projected_crs: str,
    environment_type: str,
    environment_climate: str,
    master_bounds: tuple[float, float, float, float],
    rng: np.random.Generator | None = None,
) -> ResolvedSourcePosition:
    """Resolve source position using explicit position, polar override, or preset."""
    config.validate()
    transformer_to_projected = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )
    transformer_to_wgs84 = Transformer.from_crs(
        projected_crs, "EPSG:4326", always_xy=True
    )
    center_x, center_y = transformer_to_projected.transform(*center_point)

    if config.source_position is not None:
        source_lon, source_lat = config.source_position
        source_x, source_y = transformer_to_projected.transform(
            source_lon, source_lat
        )
    else:
        if (
            config.source_distance_m is not None
            and config.source_angle_deg is not None
        ):
            distance_m = float(config.source_distance_m)
            angle_deg = float(config.source_angle_deg) % 360.0
        else:
            if rng is None:
                rng = np.random.default_rng(config.seed)
            minimum_m, maximum_m = resolve_placement_distance_range(
                config.placement_type,
                environment_type,
                environment_climate,
            )
            distance_m = float(rng.uniform(minimum_m, maximum_m))
            angle_deg = float(rng.uniform(0.0, 360.0))

        angle_rad = np.radians(angle_deg)
        source_x = center_x + distance_m * np.sin(angle_rad)
        source_y = center_y + distance_m * np.cos(angle_rad)
        source_lon, source_lat = transformer_to_wgs84.transform(source_x, source_y)

    delta_x = float(source_x - center_x)
    delta_y = float(source_y - center_y)
    resolved_distance_m = float(np.hypot(delta_x, delta_y))
    resolved_angle_deg = float(np.degrees(np.arctan2(delta_x, delta_y)) % 360.0)

    xlarge_radius_m = (
        get_environment_radius_by_size(
            environment_type, environment_climate, "xlarge"
        )
        * 1000.0
    )
    radius_tolerance_m = max(1.0, xlarge_radius_m) * 1e-6
    if resolved_distance_m > xlarge_radius_m + radius_tolerance_m:
        raise ValueError(
            "Resolved radiation source lies outside the master xlarge circular "
            f"extent: distance={resolved_distance_m:.3f} m, "
            f"xlarge_radius={xlarge_radius_m:.3f} m."
        )

    minx, miny, maxx, maxy = master_bounds
    bounds_tolerance_m = max(1.0, xlarge_radius_m) * 1e-6
    if not (
        minx - bounds_tolerance_m
        <= source_x
        <= maxx + bounds_tolerance_m
        and miny - bounds_tolerance_m
        <= source_y
        <= maxy + bounds_tolerance_m
    ):
        raise ValueError(
            "Resolved radiation source lies outside the master raster bounds."
        )

    return ResolvedSourcePosition(
        position_wgs84=(float(source_lon), float(source_lat)),
        position_projected=(float(source_x), float(source_y)),
        distance_m=resolved_distance_m,
        angle_deg=resolved_angle_deg,
    )


def resolve_distributed_sigmas(
    shape_type: str,
    sigma_x_m: float | None = None,
    sigma_y_m: float | None = None,
) -> tuple[float, float]:
    """Resolve fixed-metre Gaussian scales and validate shape invariants."""
    if shape_type not in DISTRIBUTED_SHAPE_PRESETS:
        raise ValueError(f"Unsupported shape_type '{shape_type}'.")

    preset = DISTRIBUTED_SHAPE_PRESETS[shape_type]
    if shape_type == RADIATION_SHAPE_CIRCULAR:
        if sigma_x_m is not None and sigma_y_m is None:
            sigma_y_m = sigma_x_m
        elif sigma_y_m is not None and sigma_x_m is None:
            sigma_x_m = sigma_y_m

    resolved_sigma_x = (
        preset["sigma_x_m"] if sigma_x_m is None else float(sigma_x_m)
    )
    resolved_sigma_y = (
        preset["sigma_y_m"] if sigma_y_m is None else float(sigma_y_m)
    )
    if (
        not np.isfinite(resolved_sigma_x)
        or resolved_sigma_x <= 0
        or not np.isfinite(resolved_sigma_y)
        or resolved_sigma_y <= 0
    ):
        raise ValueError("Resolved sigma_x_m and sigma_y_m must be finite and positive.")
    if (
        shape_type == RADIATION_SHAPE_CIRCULAR
        and not np.isclose(resolved_sigma_x, resolved_sigma_y)
    ):
        raise ValueError(
            "shape_type='circular' requires sigma_x_m == sigma_y_m."
        )
    if (
        shape_type == RADIATION_SHAPE_STRETCHED
        and np.isclose(resolved_sigma_x, resolved_sigma_y)
    ):
        raise ValueError(
            "shape_type='stretched' requires sigma_x_m != sigma_y_m."
        )
    return float(resolved_sigma_x), float(resolved_sigma_y)


def generate_point_source_field(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    source_position_projected: tuple[float, float],
    background_dose_rate: float,
    peak_dose_rate: float,
    reference_height_m: float,
) -> np.ndarray:
    """Generate a simplified 1 m AGL inverse-square-like benchmark field."""
    source_x, source_y = source_position_projected
    distance_squared = (
        (x_coordinates - source_x) ** 2 + (y_coordinates - source_y) ** 2
    )
    # Simplified synthetic inverse-square-like benchmark field referenced at
    # 1 m AGL; this is not a full radiation transport model.
    source_fraction = reference_height_m**2 / (
        distance_squared + reference_height_m**2
    )
    return background_dose_rate + (
        peak_dose_rate - background_dose_rate
    ) * source_fraction


def generate_distributed_source_field(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    source_position_projected: tuple[float, float],
    background_dose_rate: float,
    peak_dose_rate: float,
    sigma_x_m: float,
    sigma_y_m: float,
) -> np.ndarray:
    """Generate an axis-aligned Gaussian synthetic contamination field."""
    source_x, source_y = source_position_projected
    exponent = -0.5 * (
        ((x_coordinates - source_x) / sigma_x_m) ** 2
        + ((y_coordinates - source_y) / sigma_y_m) ** 2
    )
    return background_dose_rate + (
        peak_dose_rate - background_dose_rate
    ) * np.exp(exponent)


def apply_environment_interaction(
    radiation_field: np.ndarray,
    environment_interaction: str,
) -> np.ndarray:
    """Apply an environment interaction model; MVP supports pass-through only."""
    if environment_interaction == RADIATION_ENVIRONMENT_INTERACTION_NONE:
        return radiation_field
    raise NotImplementedError(
        "Only environment_interaction='none' is implemented in the radiation MVP."
    )


def generate_radiation_field(
    config: RadiationConfig,
    *,
    center_point: tuple[float, float],
    environment_type: str,
    environment_climate: str,
    meter_per_bin: float,
    projected_crs: str,
    xedges: np.ndarray,
    yedges: np.ndarray,
    master_shape: tuple[int, int],
    master_bounds: tuple[float, float, float, float],
) -> RadiationGenerationResult:
    """Generate one radiation scenario on an existing SAREnv master grid."""
    if not isinstance(config, RadiationConfig):
        raise TypeError("config must be an instance of RadiationConfig.")
    config.validate()
    if not np.isfinite(meter_per_bin) or meter_per_bin <= 0:
        raise ValueError("meter_per_bin must be a finite value greater than 0.")

    xedges = np.asarray(xedges, dtype=float)
    yedges = np.asarray(yedges, dtype=float)
    if xedges.ndim != 1 or yedges.ndim != 1:
        raise ValueError("xedges and yedges must be one-dimensional arrays.")
    if len(xedges) < 2 or len(yedges) < 2:
        raise ValueError("xedges and yedges must each contain at least two values.")
    if not np.isfinite(xedges).all() or not np.isfinite(yedges).all():
        raise ValueError("xedges and yedges must contain only finite values.")
    if not (np.diff(xedges) > 0).all() or not (np.diff(yedges) > 0).all():
        raise ValueError("xedges and yedges must be strictly increasing.")

    grid_shape = (len(yedges) - 1, len(xedges) - 1)
    if tuple(master_shape) != grid_shape:
        raise ValueError(
            f"Master shape {tuple(master_shape)} does not match grid shape "
            f"{grid_shape}."
        )

    background_dose_rate = resolve_background_dose_rate(
        config.background_dose_rate
    )
    peak_dose_rate = resolve_peak_dose_rate(
        config.source_type,
        config.intensity_type,
        config.peak_dose_rate,
        background_dose_rate,
    )
    rng = np.random.default_rng(config.seed)
    resolved_position = resolve_source_position(
        config=config,
        center_point=center_point,
        projected_crs=projected_crs,
        environment_type=environment_type,
        environment_climate=environment_climate,
        master_bounds=master_bounds,
        rng=rng,
    )

    sigma_x_m = None
    sigma_y_m = None
    if config.source_type == RADIATION_SOURCE_DISTRIBUTED:
        sigma_x_m, sigma_y_m = resolve_distributed_sigmas(
            config.shape_type,
            config.sigma_x_m,
            config.sigma_y_m,
        )

    source = ResolvedRadiationSource(
        source_type=config.source_type,
        position=resolved_position,
        peak_dose_rate=peak_dose_rate,
        sigma_x_m=sigma_x_m,
        sigma_y_m=sigma_y_m,
    )
    sources = (source,)

    x_centers = (xedges[:-1] + xedges[1:]) / 2.0
    y_centers = (yedges[:-1] + yedges[1:]) / 2.0
    x_coordinates, y_coordinates = np.meshgrid(x_centers, y_centers)

    if config.source_type == RADIATION_SOURCE_POINT:
        radiation_field = generate_point_source_field(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            source_position_projected=resolved_position.position_projected,
            background_dose_rate=background_dose_rate,
            peak_dose_rate=peak_dose_rate,
            reference_height_m=float(config.reference_height_m),
        )
        resolved_field_model = POINT_FIELD_MODEL_NAME
    elif config.source_type == RADIATION_SOURCE_DISTRIBUTED:
        radiation_field = generate_distributed_source_field(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            source_position_projected=resolved_position.position_projected,
            background_dose_rate=background_dose_rate,
            peak_dose_rate=peak_dose_rate,
            sigma_x_m=sigma_x_m,
            sigma_y_m=sigma_y_m,
        )
        resolved_field_model = DISTRIBUTED_FIELD_MODEL_NAME
    else:
        raise ValueError(f"Unsupported source_type '{config.source_type}'.")

    radiation_field = apply_environment_interaction(
        radiation_field, config.environment_interaction
    )
    if radiation_field.shape != tuple(master_shape):
        raise ValueError(
            f"Generated radiation shape {radiation_field.shape} does not match "
            f"master heatmap shape {tuple(master_shape)}."
        )
    if not np.isfinite(radiation_field).all():
        raise ValueError("Generated radiation field contains NaN or infinite values.")

    metadata = {
        "source_type": config.source_type,
        "placement_type": config.placement_type,
        "intensity_type": config.intensity_type,
        "shape_type": config.shape_type,
        "seed": int(config.seed),
        "quantity": GROUND_REFERENCE_QUANTITY,
        "unit": DOSE_RATE_UNIT,
        "reference_height_m": float(config.reference_height_m),
        "background_dose_rate": background_dose_rate,
        "peak_dose_rate": peak_dose_rate,
        "source_position_wgs84": list(resolved_position.position_wgs84),
        "source_position_projected": list(
            resolved_position.position_projected
        ),
        "source_distance_m": resolved_position.distance_m,
        "source_angle_deg": resolved_position.angle_deg,
        "sigma_x_m": sigma_x_m,
        "sigma_y_m": sigma_y_m,
        "source_count": config.source_count,
        "field_model": resolved_field_model,
        "field_model_requested": config.field_model,
        "environment_interaction": config.environment_interaction,
        "time_model": config.time_model,
        "measurement_model": config.measurement_model,
        "projected_crs": str(projected_crs),
        "meter_per_bin": float(meter_per_bin),
        "raster_shape": list(master_shape),
        "raster_bounds_projected": [float(value) for value in master_bounds],
    }
    return RadiationGenerationResult(
        field=radiation_field,
        metadata=metadata,
        sources=sources,
    )


__all__ = [
    "DEFAULT_BACKGROUND_DOSE_RATE",
    "DISTRIBUTED_DOSE_RATE_PRESETS",
    "DISTRIBUTED_SHAPE_PRESETS",
    "DOSE_RATE_UNIT",
    "GROUND_REFERENCE_QUANTITY",
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
    "RadiationGenerationResult",
    "ResolvedRadiationSource",
    "ResolvedSourcePosition",
    "apply_environment_interaction",
    "generate_distributed_source_field",
    "generate_point_source_field",
    "generate_radiation_field",
    "resolve_background_dose_rate",
    "resolve_distributed_sigmas",
    "resolve_peak_dose_rate",
    "resolve_placement_distance_range",
    "resolve_source_position",
]
