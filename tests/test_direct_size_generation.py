import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from sarenv.core.generation import DataGenerator, Environment
from sarenv.utils.lost_person_behavior import (
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    get_available_sizes,
    get_environment_radius_by_size,
)


CENTER_POINT = (-2.66962, 51.42351)
EXPECTED_FLAT_TEMPERATE_RADII_KM = {
    "small": 0.6,
    "medium": 1.8,
    "large": 3.2,
    "xlarge": 9.9,
}


@pytest.mark.parametrize(
    ("size", "expected_radius_km"),
    EXPECTED_FLAT_TEMPERATE_RADII_KM.items(),
)
def test_requested_size_controls_pre_query_polygon_bounds(
    monkeypatch, size, expected_radius_km
):
    """The selected size must define the polygon before Environment queries OSM."""
    generator = DataGenerator()
    sentinel_environment = object()
    monkeypatch.setattr(
        generator._builder, "build", lambda: sentinel_environment
    )

    result = generator.generate_environment(
        center_point=CENTER_POINT,
        size=size,
        environment_climate=CLIMATE_TEMPERATE,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        meter_per_bin=30,
    )

    assert result is sentinel_environment
    projected_polygon = gpd.GeoSeries(
        [generator._builder.polygon], crs="EPSG:4326"
    ).to_crs(generator._builder.projected_crs)
    minx, miny, maxx, maxy = projected_polygon.total_bounds
    expected_dimension_m = 2.0 * expected_radius_km * 1000.0
    assert maxx - minx == pytest.approx(expected_dimension_m, abs=1e-3)
    assert maxy - miny == pytest.approx(expected_dimension_m, abs=1e-3)


def test_size_names_and_physical_radii_remain_unchanged():
    assert get_available_sizes() == ["small", "medium", "large", "xlarge"]
    for size, expected_radius_km in EXPECTED_FLAT_TEMPERATE_RADII_KM.items():
        assert get_environment_radius_by_size(
            ENVIRONMENT_TYPE_FLAT, CLIMATE_TEMPERATE, size
        ) == pytest.approx(expected_radius_km)


def test_direct_small_uses_expected_40_by_40_raster(monkeypatch):
    """A 1.2 km square at 30 m must not lose a bin to projection round-off."""
    generator = DataGenerator()
    polygon = generator._create_circular_polygon(*CENTER_POINT, 0.6)
    monkeypatch.setattr(Environment, "_load_features", lambda self: None)

    environment = Environment(
        bounding_polygon=polygon,
        sample_distance=1,
        meter_per_bin=30,
        buffer_val=0,
        tags={},
        projected_crs="EPSG:32630",
    )

    assert len(environment.xedges) - 1 == 40
    assert len(environment.yedges) - 1 == 40


class _FakeSmallEnvironment:
    def __init__(self):
        projected_center = gpd.GeoSeries.from_xy(
            [CENTER_POINT[0]], [CENTER_POINT[1]], crs="EPSG:4326"
        ).to_crs("EPSG:32630")
        center_x = projected_center.x.iloc[0]
        center_y = projected_center.y.iloc[0]

        self.projected_crs = "EPSG:32630"
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
                            (center_x - 30.0, center_y - 30.0),
                            (center_x + 30.0, center_y + 30.0),
                        ]
                    )
                ],
                crs=self.projected_crs,
            )
        }

    def get_combined_heatmap(self):
        return np.ones((40, 40), dtype=float)


def test_export_dataset_defaults_to_direct_small_without_larger_generation(
    tmp_path, monkeypatch
):
    generator = DataGenerator()
    requested_sizes = []

    def fake_generate_environment(
        center_point, size, environment_climate, environment_type, meter_per_bin
    ):
        requested_sizes.append(size)
        return _FakeSmallEnvironment()

    monkeypatch.setattr(
        generator, "generate_environment", fake_generate_environment
    )
    output_directory = tmp_path / "direct_small"

    generator.export_dataset(
        center_point=CENTER_POINT,
        output_directory=str(output_directory),
        environment_type=ENVIRONMENT_TYPE_FLAT,
        environment_climate=CLIMATE_TEMPERATE,
        meter_per_bin=30,
    )

    assert requested_sizes == ["small"]
    heatmap = np.load(output_directory / "heatmap.npy", allow_pickle=False)
    metadata = json.loads(
        (output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    features = json.loads(
        (output_directory / "features.geojson").read_text(encoding="utf-8")
    )

    assert heatmap.shape == (40, 40)
    assert heatmap.sum() == pytest.approx(1.0)
    assert metadata["environment_size"] == "small"
    assert metadata["radius_km"] == pytest.approx(0.6)
    assert metadata["meter_per_bin"] == pytest.approx(30.0)
    assert metadata["raster_shape"] == [40, 40]
    assert metadata["bounds_projected"][2] - metadata["bounds_projected"][0] == pytest.approx(
        1200.0
    )
    assert metadata["bounds_projected"][3] - metadata["bounds_projected"][1] == pytest.approx(
        1200.0
    )
    assert features["environment_size"] == "small"
    assert features["radius_km"] == pytest.approx(0.6)
    assert len(features["features"]) == 1


def test_invalid_export_size_fails_before_environment_generation(
    tmp_path, monkeypatch
):
    generator = DataGenerator()

    def fail_if_called(*args, **kwargs):
        pytest.fail("Environment generation must not start for an invalid size.")

    monkeypatch.setattr(generator, "generate_environment", fail_if_called)
    with pytest.raises(ValueError, match="Invalid size"):
        generator.export_dataset(
            center_point=CENTER_POINT,
            output_directory=str(tmp_path / "invalid"),
            environment_type=ENVIRONMENT_TYPE_FLAT,
            environment_climate=CLIMATE_TEMPERATE,
            meter_per_bin=30,
            target_size="extra_large",
        )
