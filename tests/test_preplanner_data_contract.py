"""Round-trip contracts for the active pipeline before path planning."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from sarenv import DataGenerator, DatasetLoader, LostPersonLocationGenerator
from sarenv.io.lost_person import (
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
from sarenv.radiation.surface import (
    MEDIUM_ACTIVITY_DENSITY_BQ_M2,
)


CENTER_POINT = (-2.66962, 51.42351)
PROJECTED_CRS = "EPSG:32630"


class _OfflineSmallEnvironment:
    """One local feature fixture; no OSM or radiation behavior is replaced."""

    def __init__(self):
        center = gpd.GeoSeries.from_xy(
            [CENTER_POINT[0]], [CENTER_POINT[1]], crs="EPSG:4326"
        ).to_crs(PROJECTED_CRS)
        center_x = center.x.iloc[0]
        center_y = center.y.iloc[0]
        self.projected_crs = PROJECTED_CRS
        self.minx = center_x - 600.0
        self.miny = center_y - 600.0
        self.maxx = center_x + 600.0
        self.maxy = center_y + 600.0
        self.xedges = np.linspace(self.minx, self.maxx, 41)
        self.yedges = np.linspace(self.miny, self.maxy, 41)
        self.heatmaps = {
            "structure": np.ones((40, 40), dtype=np.uint8)
        }
        self.features = {
            "structure": gpd.GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            (center_x - 60.0, center_y),
                            (center_x + 60.0, center_y),
                        ]
                    )
                ],
                crs=PROJECTED_CRS,
            )
        }

    def get_combined_heatmap(self):
        return np.ones((40, 40), dtype=np.float64)


@pytest.fixture
def direct_small_dataset(tmp_path, monkeypatch):
    generator = DataGenerator()
    monkeypatch.setattr(
        generator,
        "generate_environment",
        lambda *args, **kwargs: _OfflineSmallEnvironment(),
    )
    directory = tmp_path / "small_base"
    generator.export_dataset(
        center_point=CENTER_POINT,
        output_directory=str(directory),
        environment_type="flat",
        environment_climate="temperate",
        meter_per_bin=30,
        target_size="small",
    )
    return directory


def test_direct_small_sar_round_trip_uses_authoritative_metadata(
    direct_small_dataset,
):
    saved_heatmap = np.load(
        direct_small_dataset / "heatmap.npy", allow_pickle=False
    )
    loader = DatasetLoader(str(direct_small_dataset))
    item = loader.load_environment()

    assert loader.dataset_size == "small"
    assert item.size == "small"
    assert item.radius_km == pytest.approx(0.6)
    assert item.projected_crs == PROJECTED_CRS
    assert item.meter_per_bin == pytest.approx(30.0)
    assert item.bounds[2] - item.bounds[0] == pytest.approx(1200.0)
    assert item.bounds[3] - item.bounds[1] == pytest.approx(1200.0)
    assert item.heatmap.shape == (40, 40)
    assert np.array_equal(item.heatmap, saved_heatmap)
    assert item.heatmap.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="physically smaller"):
        loader.load_environment("large")


def test_loader_rejects_conflicting_duplicate_metadata(direct_small_dataset):
    features_path = direct_small_dataset / "features.geojson"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    features["environment_size"] = "xlarge"
    features_path.write_text(json.dumps(features), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        DatasetLoader(str(direct_small_dataset)).load_environment()


def test_loader_rejects_inconsistent_raster_extent(direct_small_dataset):
    metadata_path = direct_small_dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["bounds_projected"][2] += 30.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    features_path = direct_small_dataset / "features.geojson"
    features = json.loads(features_path.read_text(encoding="utf-8"))
    features["bounds"][2] += 30.0
    features_path.write_text(json.dumps(features), encoding="utf-8")

    with pytest.raises(ValueError, match="different raster extents"):
        DatasetLoader(str(direct_small_dataset)).load_environment()


def test_lost_person_round_trip_preserves_projected_coordinates(
    direct_small_dataset,
):
    item = DatasetLoader(str(direct_small_dataset)).load_environment()
    random.seed(42)
    locations = LostPersonLocationGenerator(item).generate_locations(8, 0)
    output_path = save_lost_person_locations(
        locations, item, direct_small_dataset / "lost_person"
    )
    loaded = load_lost_person_locations(output_path)

    assert len(loaded.points) == len(locations) == 8
    assert loaded.environment_size == "small"
    assert loaded.projected_crs == item.projected_crs
    assert loaded.bounds == pytest.approx(item.bounds)
    assert [point.coords[0] for point in loaded.points] == [
        point.coords[0] for point in locations
    ]
    minx, miny, maxx, maxy = item.bounds
    assert all(
        minx <= point.x <= maxx and miny <= point.y <= maxy
        for point in loaded.points
    )


def _surface_result_for_item(item):
    center = gpd.GeoSeries.from_xy(
        [item.center_point[0]], [item.center_point[1]], crs="EPSG:4326"
    ).to_crs(item.projected_crs)
    center_x = center.x.iloc[0]
    center_y = center.y.iloc[0]
    grid = GridSpec.from_bounds(
        (center_x - 15.0, center_y - 15.0, center_x + 15.0, center_y + 15.0),
        item.projected_crs,
    )
    source = UniformPolygonSource(
        UniformPolygonConfig(
            source_id="surface_only_test",
            geometry=box(
                center_x - 5.0,
                center_y - 5.0,
                center_x + 5.0,
                center_y + 5.0,
            ),
            activity_density_bq_m2=MEDIUM_ACTIVITY_DENSITY_BQ_M2,
            crs=item.projected_crs,
        )
    )
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(), grid
    )
    return simulate_surface_source(source, kernel, nominal_grid=grid)


def test_physical_surface_round_trip_preserves_every_quantity(
    direct_small_dataset,
):
    item = DatasetLoader(str(direct_small_dataset)).load_environment()
    result = _surface_result_for_item(item)
    paths = save_surface_simulation(
        result, direct_small_dataset / "radiation" / "surface_only"
    )
    loaded = load_surface_simulation(paths.metadata.parent)

    assert paths.activity_density.name == "activity_density_bq_m2.npy"
    assert paths.cell_activity.name == "cell_activity_bq.npy"
    assert paths.photon_fluence_rate.name == "photon_fluence_rate_ph_m2_s.npy"
    assert paths.collision_air_kerma_rate.name == (
        "collision_air_kerma_rate_uGy_h.npy"
    )
    assert np.array_equal(
        loaded.activity_density_bq_m2, result.activity_density_bq_m2
    )
    assert np.array_equal(loaded.cell_activity_bq, result.cell_activity_bq)
    assert np.array_equal(
        loaded.photon_fluence_rate_patch.photon_fluence_rate,
        result.photon_fluence_rate_patch.photon_fluence_rate,
    )
    assert np.array_equal(
        loaded.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h,
        result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h,
    )
    assert loaded.grid == result.grid
    assert loaded.metadata["scenario_type"] == "surface_only"
    assert loaded.metadata["quantities"]["activity_density_bq_m2"]["unit"] == (
        "Bq/m^2"
    )
    assert loaded.metadata["quantities"]["photon_fluence_rate_ph_m2_s"][
        "unit"
    ] == "photons/m^2/s"
    assert loaded.metadata["quantities"]["collision_air_kerma_rate_uGy_h"][
        "unit"
    ] == "uGy/h"


def test_surface_loader_rejects_silent_unit_conversion(
    direct_small_dataset,
):
    item = DatasetLoader(str(direct_small_dataset)).load_environment()
    result = _surface_result_for_item(item)
    paths = save_surface_simulation(result, direct_small_dataset / "surface")
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata["quantities"]["collision_air_kerma_rate_uGy_h"]["unit"] = "uSv/h"
    paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="unit"):
        load_surface_simulation(paths.metadata.parent)


def test_point_only_round_trip_and_active_scenario_selection(
    direct_small_dataset,
):
    item = DatasetLoader(str(direct_small_dataset)).load_environment()
    center = gpd.GeoSeries.from_xy(
        [item.center_point[0]], [item.center_point[1]], crs="EPSG:4326"
    ).to_crs(item.projected_crs)
    source = PointSource(
        PointSourceConfig(
            source_id="point_only_test",
            x_m=float(center.x.iloc[0]),
            y_m=float(center.y.iloc[0]),
            reference_excess_uSv_h=20_000.0,
            crs=item.projected_crs,
        )
    )
    path = save_point_source(
        source, direct_small_dataset / "radiation" / "point_source_meta.json"
    )
    loaded = load_point_source(path)
    assert loaded == source
    assert resolve_radiation_scenario("no_radiation") is (
        RadiationScenarioType.NO_RADIATION
    )
    assert resolve_radiation_scenario("point_only") is (
        RadiationScenarioType.POINT_ONLY
    )
    assert resolve_radiation_scenario("surface_only") is (
        RadiationScenarioType.SURFACE_ONLY
    )
    with pytest.raises(ValueError, match=r"Point \+ Surface"):
        resolve_radiation_scenario("point_plus_surface")


def test_preplanner_integration_script_stops_after_surface_round_trip(
    direct_small_dataset,
    tmp_path,
):
    script_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "sarenv_dataset"
        / "1.2_preplanner_data_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "sarenv_preplanner_example", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    item, lost_people, surface = module.run_preplanner_data_pipeline(
        direct_small_dataset,
        tmp_path / "preplanner_output",
        RadiationScenarioType.SURFACE_ONLY,
    )

    assert item.size == "small"
    assert item.heatmap.sum() == pytest.approx(1.0)
    assert len(lost_people.points) == module.NUM_LOST_PERSONS
    assert surface.metadata["scenario_type"] == "surface_only"
    assert not (tmp_path / "preplanner_output" / "point_only").exists()
