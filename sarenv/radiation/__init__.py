"""Independent deterministic synthetic-radiation models for SAREnv geography."""

from .common.grid import GridSpec, RADIATION_RESOLUTION_M
from .common.io import (
    LoadedSurfaceSimulation,
    SavedSurfacePaths,
    load_point_source,
    load_surface_simulation,
    save_point_source,
    save_surface_simulation,
)
from .common.terrain_context import TerrainContext
from .composite.field import CompositeRadiationField
from .point.config import PointSourceConfig
from .point.source import PointSource
from .scenario import RadiationScenarioType, resolve_radiation_scenario
from .surface.config import (
    BenchmarkSurfaceResponseConfig,
    SurfacePhotonResponseConfig,
    SurfaceZone,
    UniformPolygonConfig,
    ZonedPolygonConfig,
)
from .surface.dose_patch import (
    CollisionAirKermaRatePatch,
    DoseRatePatch,
    PhotonFluenceRatePatch,
)
from .surface.response_kernel import (
    BenchmarkSurfaceResponseKernel,
    SurfacePhotonResponseKernel,
)
from .surface.simulator import SurfaceSimulationResult, simulate_surface_source
from .surface.uniform_polygon import UniformPolygonSource
from .surface.zoned_polygon import ZonedPolygonSource

__all__ = [
    "BenchmarkSurfaceResponseConfig",
    "BenchmarkSurfaceResponseKernel",
    "CompositeRadiationField",
    "CollisionAirKermaRatePatch",
    "DoseRatePatch",
    "GridSpec",
    "LoadedSurfaceSimulation",
    "PointSource",
    "PointSourceConfig",
    "PhotonFluenceRatePatch",
    "RadiationScenarioType",
    "RADIATION_RESOLUTION_M",
    "SavedSurfacePaths",
    "SurfaceSimulationResult",
    "SurfacePhotonResponseConfig",
    "SurfacePhotonResponseKernel",
    "SurfaceZone",
    "TerrainContext",
    "UniformPolygonConfig",
    "UniformPolygonSource",
    "ZonedPolygonConfig",
    "ZonedPolygonSource",
    "load_point_source",
    "load_surface_simulation",
    "save_point_source",
    "save_surface_simulation",
    "resolve_radiation_scenario",
    "simulate_surface_source",
]
