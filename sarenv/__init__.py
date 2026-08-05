# sarenv/__init__.py
"""
SAREnv: A Python toolkit for generating, loading, and evaluating
Search and Rescue environment data.
"""
from .analytics.evaluator import ComparativeEvaluator
from .core.generation import DataGenerator
from .core.loading import DatasetLoader, SARDatasetItem
from .core.lost_person import LostPersonLocationGenerator
from .core.radiation import (
    DEFAULT_BACKGROUND_DOSE_RATE,
    DISTRIBUTED_DOSE_RATE_PRESETS,
    DISTRIBUTED_SHAPE_PRESETS,
    PLACEMENT_DISTANCE_PRESETS,
    POINT_DOSE_RATE_PRESETS,
    RADIATION_INTENSITY_HIGH,
    RADIATION_INTENSITY_LOW,
    RADIATION_INTENSITY_MEDIUM,
    RADIATION_PLACEMENT_FAR,
    RADIATION_PLACEMENT_MIDDLE,
    RADIATION_PLACEMENT_NEAR,
    RADIATION_SHAPE_CIRCULAR,
    RADIATION_SHAPE_STRETCHED,
    RADIATION_SOURCE_DISTRIBUTED,
    RADIATION_SOURCE_POINT,
    RadiationConfig,
    generate_radiation_field,
)
from .io.radiation_layer import generate_radiation_layer
from .utils.logging_setup import get_logger
from .utils.lost_person_behavior import (
    FEATURE_PROBABILITIES,
    CLIMATE_TEMPERATE,
    CLIMATE_DRY,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
)
from .utils.plot import visualize_heatmap, visualize_features

__all__ = [
    "ComparativeEvaluator",
    "DataGenerator",
    "DatasetLoader",
    "SARDatasetItem",
    "RadiationConfig",
    "LostPersonLocationGenerator",
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
