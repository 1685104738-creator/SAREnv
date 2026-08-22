# sarenv/__init__.py
"""
SAREnv: A Python toolkit for generating, loading, and evaluating
Search and Rescue environment data.
"""
from .analytics.evaluator import ComparativeEvaluator
from .core.generation import DataGenerator
from .core.loading import DatasetLoader, SARDatasetItem
from .core.lost_person import LostPersonLocationGenerator
from .io.lost_person import (
    LostPersonLocations,
    load_lost_person_locations,
    save_lost_person_locations,
)
from .utils.logging_setup import get_logger
from .utils.lost_person_behavior import (
    FEATURE_PROBABILITIES,
    CLIMATE_TEMPERATE,
    CLIMATE_DRY,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
)
from .utils.plot import visualize_heatmap, visualize_features


_LEGACY_RADIATION_CORE_NAMES = {
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
}


def __getattr__(name: str):
    """Lazily expose deprecated radiation names without coupling base SAR imports."""
    if name in _LEGACY_RADIATION_CORE_NAMES:
        from .core import radiation as legacy_radiation

        return getattr(legacy_radiation, name)
    if name == "generate_radiation_layer":
        from .io.radiation_layer import generate_radiation_layer

        return generate_radiation_layer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ComparativeEvaluator",
    "DataGenerator",
    "DatasetLoader",
    "SARDatasetItem",
    "RadiationConfig",
    "LostPersonLocationGenerator",
    "LostPersonLocations",
    "load_lost_person_locations",
    "save_lost_person_locations",
    "get_logger",
    "ENVIRONMENT_TYPE_FLAT",
    "ENVIRONMENT_TYPE_MOUNTAINOUS",
    "CLIMATE_DRY",
    "CLIMATE_TEMPERATE",
    "FEATURE_PROBABILITIES",
    "DEFAULT_BACKGROUND_DOSE_RATE",
    "POINT_DOSE_RATE_PRESETS",
    "DISTRIBUTED_DOSE_RATE_PRESETS",
    "DISTRIBUTED_SHAPE_PRESETS",
    "PLACEMENT_DISTANCE_PRESETS",
    "RADIATION_SOURCE_POINT",
    "RADIATION_SOURCE_DISTRIBUTED",
    "RADIATION_PLACEMENT_NEAR",
    "RADIATION_PLACEMENT_MIDDLE",
    "RADIATION_PLACEMENT_FAR",
    "RADIATION_INTENSITY_LOW",
    "RADIATION_INTENSITY_MEDIUM",
    "RADIATION_INTENSITY_HIGH",
    "RADIATION_SHAPE_CIRCULAR",
    "RADIATION_SHAPE_STRETCHED",
    "generate_radiation_field",
    "generate_radiation_layer",
    "visualize_heatmap",
    "visualize_features",
]
