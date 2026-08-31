"""Offline evaluation APIs for completed SAR missions."""

from .evaluator import (
    EvaluationConfig,
    PostRunEvaluationResult,
    PostRunEvaluator,
    evaluate_native_sar_metrics,
)
from .exposure import (
    DEFAULT_USV_CONTRACT,
    RadiationQuantityContract,
    RouteIntegrationSamples,
    RouteRadiationIntegral,
    SurvivorRadiationRecord,
    dose_from_distance_integral,
    evaluate_survivor_radiation,
    integrate_radiation_along_route,
    prepare_route_integration_samples,
    summarise_survivor_radiation,
)
from .surface_truth import SurfaceKermaTruthField
from .trajectory import (
    ExecutedTrajectory,
    SurvivorDiscovery,
    TrajectoryNode,
    calculate_turn_diagnostics,
    discover_survivors,
    first_discovery_distance_m,
    load_executed_trajectory,
)

__all__ = [
    "EvaluationConfig",
    "DEFAULT_USV_CONTRACT",
    "ExecutedTrajectory",
    "PostRunEvaluationResult",
    "PostRunEvaluator",
    "RadiationQuantityContract",
    "RouteIntegrationSamples",
    "RouteRadiationIntegral",
    "SurvivorDiscovery",
    "SurvivorRadiationRecord",
    "SurfaceKermaTruthField",
    "TrajectoryNode",
    "calculate_turn_diagnostics",
    "discover_survivors",
    "dose_from_distance_integral",
    "evaluate_native_sar_metrics",
    "evaluate_survivor_radiation",
    "first_discovery_distance_m",
    "integrate_radiation_along_route",
    "prepare_route_integration_samples",
    "load_executed_trajectory",
    "summarise_survivor_radiation",
]
