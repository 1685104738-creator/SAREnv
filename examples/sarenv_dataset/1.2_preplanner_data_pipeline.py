"""Verify the saved Small SAR, lost-person, and one radiation scenario."""

from __future__ import annotations

from pathlib import Path
import random

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sarenv import (
    DatasetLoader,
    LostPersonLocationGenerator,
    get_logger,
    load_lost_person_locations,
    save_lost_person_locations,
)
from sarenv.radiation import (
    GridSpec,
    PointSource,
    PointSourceConfig,
    RadiationScenarioType,
    SurfacePhotonResponseConfig,
    SurfacePhotonResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
    load_point_source,
    load_surface_simulation,
    resolve_radiation_scenario,
    save_point_source,
    save_surface_simulation,
    simulate_surface_source,
)
from sarenv.radiation.surface import MEDIUM_ACTIVITY_DENSITY_BQ_M2


log = get_logger()
EXAMPLE_DIRECTORY = Path(__file__).resolve().parent
DATASET_DIRECTORY = (
    EXAMPLE_DIRECTORY / "sarenv_outputs" / "radiation_area_01"
)
OUTPUT_DIRECTORY = DATASET_DIRECTORY / "preplanner_outputs"
RADIATION_SCENARIO = RadiationScenarioType.SURFACE_ONLY
NUM_LOST_PERSONS = 10


def _projected_center(item) -> tuple[float, float]:
    center = gpd.GeoSeries.from_xy(
        [item.center_point[0]], [item.center_point[1]], crs="EPSG:4326"
    ).to_crs(item.projected_crs)
    return float(center.x.iloc[0]), float(center.y.iloc[0])


def _run_surface_only(item, output_directory: Path):
    center_x, center_y = _projected_center(item)
    grid = GridSpec.from_bounds(
        (center_x - 25.0, center_y - 25.0, center_x + 25.0, center_y + 25.0),
        item.projected_crs,
    )
    source = UniformPolygonSource(
        UniformPolygonConfig(
            source_id="preplanner_surface_only",
            geometry=box(
                center_x - 10.0,
                center_y - 10.0,
                center_x + 10.0,
                center_y + 10.0,
            ),
            activity_density_bq_m2=MEDIUM_ACTIVITY_DENSITY_BQ_M2,
            crs=item.projected_crs,
        )
    )
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(), grid
    )
    result = simulate_surface_source(source, kernel, nominal_grid=grid)
    paths = save_surface_simulation(result, output_directory / "surface_only")
    loaded = load_surface_simulation(paths.metadata.parent)
    arrays_match = all(
        (
            np.array_equal(
                loaded.activity_density_bq_m2,
                result.activity_density_bq_m2,
            ),
            np.array_equal(
                loaded.cell_activity_bq,
                result.cell_activity_bq,
            ),
            np.array_equal(
                loaded.photon_fluence_rate_patch.photon_fluence_rate,
                result.photon_fluence_rate_patch.photon_fluence_rate,
            ),
            np.array_equal(
                loaded.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h,
                result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h,
            ),
        )
    )
    if not arrays_match or loaded.grid != result.grid:
        raise ValueError("Surface radiation save/load verification failed.")
    return loaded


def _run_point_only(item, output_directory: Path):
    center_x, center_y = _projected_center(item)
    source = PointSource(
        PointSourceConfig(
            source_id="preplanner_point_only",
            x_m=center_x,
            y_m=center_y,
            reference_excess_uSv_h=20_000.0,
            crs=item.projected_crs,
        )
    )
    path = save_point_source(source, output_directory / "point_source_meta.json")
    loaded = load_point_source(path)
    if loaded != source:
        raise ValueError("Point radiation save/load verification failed.")
    return loaded


def run_preplanner_data_pipeline(
    dataset_directory: Path = DATASET_DIRECTORY,
    output_directory: Path = OUTPUT_DIRECTORY,
    radiation_scenario: RadiationScenarioType | str = RADIATION_SCENARIO,
):
    """Run saved Small data through lost-person and one radiation branch."""
    scenario = resolve_radiation_scenario(radiation_scenario)
    loader = DatasetLoader(str(dataset_directory))
    item = loader.load_environment()
    if item.size != "small":
        raise ValueError(
            "This integration example requires a directly generated Small dataset."
        )
    if not np.isclose(item.heatmap.sum(), 1.0, atol=1e-6):
        raise ValueError("Small probability heatmap must sum to 1.0.")

    random.seed(42)
    locations = LostPersonLocationGenerator(item).generate_locations(
        NUM_LOST_PERSONS, 0
    )
    lost_path = save_lost_person_locations(
        locations, item, output_directory / "lost_person"
    )
    loaded_locations = load_lost_person_locations(lost_path)
    original_coordinates = tuple((point.x, point.y) for point in locations)
    restored_coordinates = tuple(
        (point.x, point.y) for point in loaded_locations.points
    )
    lost_metadata_matches = (
        loaded_locations.environment_size == item.size
        and loaded_locations.bounds == item.bounds
        and loaded_locations.projected_crs == item.projected_crs
    )
    if (
        len(loaded_locations.points) != NUM_LOST_PERSONS
        or restored_coordinates != original_coordinates
        or not lost_metadata_matches
    ):
        raise ValueError("Lost-person save/load verification failed.")

    radiation_result = None
    if scenario is RadiationScenarioType.POINT_ONLY:
        radiation_result = _run_point_only(item, output_directory / "point_only")
    elif scenario is RadiationScenarioType.SURFACE_ONLY:
        radiation_result = _run_surface_only(item, output_directory)

    log.info(
        "Pre-planner pipeline PASS: size=%s, heatmap=%s, probability=%.12f, "
        "lost_persons=%d, radiation=%s",
        item.size,
        item.heatmap.shape,
        item.heatmap.sum(),
        len(loaded_locations.points),
        scenario.value,
    )
    return item, loaded_locations, radiation_result


if __name__ == "__main__":
    run_preplanner_data_pipeline()
