"""Tests for the optional synthetic radiation layer."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from sarenv import (
    CLIMATE_TEMPERATE,
    ENVIRONMENT_TYPE_FLAT,
    DataGenerator,
    DatasetLoader,
    RadiationConfig,
)
from sarenv.core.radiation import (
    DEFAULT_BACKGROUND_DOSE_RATE,
    DISTRIBUTED_DOSE_RATE_PRESETS,
    POINT_DOSE_RATE_PRESETS,
    RADIATION_PLACEMENT_FAR,
    RADIATION_PLACEMENT_MIDDLE,
    RADIATION_PLACEMENT_NEAR,
    RADIATION_SHAPE_CIRCULAR,
    RADIATION_SHAPE_STRETCHED,
    RADIATION_SOURCE_DISTRIBUTED,
    RADIATION_SOURCE_POINT,
    generate_radiation_field,
    resolve_distributed_sigmas,
    resolve_placement_distance_range,
)
from sarenv.utils.lost_person_behavior import get_environment_radius_by_size
from shapely.geometry import LineString

CENTER_POINT = (-2.66962, 51.42351)
PROJECTED_CRS = "EPSG:32630"


def _projected_center() -> tuple[float, float]:
    transformer = Transformer.from_crs(
        "EPSG:4326", PROJECTED_CRS, always_xy=True
    )
    return transformer.transform(*CENTER_POINT)


def _make_grid(
    meter_per_bin: float = 100.0,
    half_width_m: float = 9_900.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[float, ...]]:
    center_x, center_y = _projected_center()
    bin_count = int((2 * half_width_m) / meter_per_bin)
    xedges = np.linspace(
        center_x - half_width_m, center_x + half_width_m, bin_count + 1
    )
    yedges = np.linspace(
        center_y - half_width_m, center_y + half_width_m, bin_count + 1
    )
    shape = (bin_count, bin_count)
    bounds = (
        center_x - half_width_m,
        center_y - half_width_m,
        center_x + half_width_m,
        center_y + half_width_m,
    )
    return xedges, yedges, shape, bounds


def _generate(
    config: RadiationConfig,
    *,
    meter_per_bin: float = 100.0,
    half_width_m: float = 9_900.0,
):
    xedges, yedges, shape, bounds = _make_grid(
        meter_per_bin=meter_per_bin,
        half_width_m=half_width_m,
    )
    return generate_radiation_field(
        config,
        center_point=CENTER_POINT,
        environment_type=ENVIRONMENT_TYPE_FLAT,
        environment_climate=CLIMATE_TEMPERATE,
        meter_per_bin=meter_per_bin,
        projected_crs=PROJECTED_CRS,
        xedges=xedges,
        yedges=yedges,
        master_shape=shape,
        master_bounds=bounds,
    )


def test_default_point_field_uses_unscaled_reference_height_formula():
    result = _generate(
        RadiationConfig(source_distance_m=0.0, source_angle_deg=0.0),
        meter_per_bin=100.0,
        half_width_m=1_050.0,
    )

    center_index = result.field.shape[0] // 2
    peak = POINT_DOSE_RATE_PRESETS["medium"]
    background = DEFAULT_BACKGROUND_DOSE_RATE
    assert result.field[center_index, center_index] == pytest.approx(peak)

    expected_at_100_m = background + (peak - background) / (100.0**2 + 1.0)
    assert result.field[center_index, center_index + 1] == pytest.approx(
        expected_at_100_m
    )
    assert result.metadata["field_model"] == "inverse_square_like"
    assert result.metadata["quantity"] == "ground_reference_gamma_dose_rate"
    assert result.metadata["unit"] == "uSv/h"
    assert result.metadata["reference_height_m"] == 1.0
    assert np.isfinite(result.field).all()


def test_off_grid_point_source_is_not_normalized_to_theoretical_peak():
    result = _generate(
        RadiationConfig(
            source_distance_m=37.0,
            source_angle_deg=90.0,
        ),
        meter_per_bin=100.0,
        half_width_m=1_050.0,
    )
    center_index = result.field.shape[0] // 2
    peak = POINT_DOSE_RATE_PRESETS["medium"]
    background = DEFAULT_BACKGROUND_DOSE_RATE
    expected_nearest_value = background + (
        (peak - background) / (37.0**2 + 1.0)
    )

    assert result.field.max() < peak
    assert result.field[center_index, center_index] == pytest.approx(
        expected_nearest_value
    )
    assert result.field[0, 0] == pytest.approx(background, abs=0.02)


@pytest.mark.parametrize(
    ("source_type", "expected_peak"),
    [
        (RADIATION_SOURCE_POINT, POINT_DOSE_RATE_PRESETS["medium"]),
        (
            RADIATION_SOURCE_DISTRIBUTED,
            DISTRIBUTED_DOSE_RATE_PRESETS["medium"],
        ),
    ],
)
def test_intensity_presets_are_source_type_specific(
    source_type, expected_peak
):
    result = _generate(
        RadiationConfig(
            source_type=source_type,
            source_distance_m=0.0,
            source_angle_deg=0.0,
        ),
        meter_per_bin=100.0,
        half_width_m=1_050.0,
    )
    assert result.metadata["peak_dose_rate"] == expected_peak


def test_explicit_background_and_peak_override_presets():
    result = _generate(
        RadiationConfig(
            background_dose_rate=0.35,
            peak_dose_rate=500_000.0,
            source_distance_m=0.0,
            source_angle_deg=0.0,
        ),
        meter_per_bin=100.0,
        half_width_m=1_050.0,
    )
    assert result.metadata["background_dose_rate"] == 0.35
    assert result.metadata["peak_dose_rate"] == 500_000.0


def test_distributed_circular_field_is_isotropic():
    result = _generate(
        RadiationConfig(
            source_type=RADIATION_SOURCE_DISTRIBUTED,
            shape_type=RADIATION_SHAPE_CIRCULAR,
            source_distance_m=0.0,
            source_angle_deg=0.0,
        ),
        meter_per_bin=50.0,
        half_width_m=525.0,
    )
    center = result.field.shape[0] // 2
    assert result.field[center, center + 2] == pytest.approx(
        result.field[center + 2, center]
    )
    assert result.metadata["sigma_x_m"] == 150.0
    assert result.metadata["sigma_y_m"] == 150.0
    assert result.metadata["field_model"] == "gaussian"


def test_distributed_stretched_field_is_elongated_along_easting():
    result = _generate(
        RadiationConfig(
            source_type=RADIATION_SOURCE_DISTRIBUTED,
            shape_type=RADIATION_SHAPE_STRETCHED,
            source_distance_m=0.0,
            source_angle_deg=0.0,
        ),
        meter_per_bin=50.0,
        half_width_m=525.0,
    )
    center = result.field.shape[0] // 2
    assert result.field[center, center + 4] > result.field[center + 4, center]
    assert result.metadata["sigma_x_m"] == 300.0
    assert result.metadata["sigma_y_m"] == 100.0


def test_distributed_sigma_overrides_preserve_shape_contracts():
    assert resolve_distributed_sigmas(
        RADIATION_SHAPE_CIRCULAR, sigma_x_m=225.0
    ) == (225.0, 225.0)
    assert resolve_distributed_sigmas(
        RADIATION_SHAPE_STRETCHED,
        sigma_x_m=450.0,
        sigma_y_m=125.0,
    ) == (450.0, 125.0)

    with pytest.raises(ValueError, match="circular"):
        resolve_distributed_sigmas(
            RADIATION_SHAPE_CIRCULAR,
            sigma_x_m=200.0,
            sigma_y_m=100.0,
        )


@pytest.mark.parametrize(
    "placement_type",
    [
        RADIATION_PLACEMENT_NEAR,
        RADIATION_PLACEMENT_MIDDLE,
        RADIATION_PLACEMENT_FAR,
    ],
)
def test_automatic_placement_stays_within_central_preset(
    placement_type,
):
    minimum_m, maximum_m = resolve_placement_distance_range(
        placement_type,
        ENVIRONMENT_TYPE_FLAT,
        CLIMATE_TEMPERATE,
    )
    result = _generate(RadiationConfig(placement_type=placement_type, seed=42))
    assert minimum_m <= result.metadata["source_distance_m"] <= maximum_m


def test_placement_rules_match_confirmed_search_radius_ranges():
    small = get_environment_radius_by_size(
        ENVIRONMENT_TYPE_FLAT, CLIMATE_TEMPERATE, "small"
    )
    medium = get_environment_radius_by_size(
        ENVIRONMENT_TYPE_FLAT, CLIMATE_TEMPERATE, "medium"
    )
    large = get_environment_radius_by_size(
        ENVIRONMENT_TYPE_FLAT, CLIMATE_TEMPERATE, "large"
    )
    assert resolve_placement_distance_range(
        RADIATION_PLACEMENT_NEAR,
        ENVIRONMENT_TYPE_FLAT,
        CLIMATE_TEMPERATE,
    ) == pytest.approx((0.25 * small * 1000, 0.75 * small * 1000))
    assert resolve_placement_distance_range(
        RADIATION_PLACEMENT_MIDDLE,
        ENVIRONMENT_TYPE_FLAT,
        CLIMATE_TEMPERATE,
    ) == pytest.approx((small * 1000, medium * 1000))
    assert resolve_placement_distance_range(
        RADIATION_PLACEMENT_FAR,
        ENVIRONMENT_TYPE_FLAT,
        CLIMATE_TEMPERATE,
    ) == pytest.approx((medium * 1000, 0.90 * large * 1000))


def test_seed_reproducibility_and_different_seed_variation():
    config = RadiationConfig(
        source_type=RADIATION_SOURCE_DISTRIBUTED,
        placement_type=RADIATION_PLACEMENT_MIDDLE,
        seed=42,
    )
    first = _generate(config)
    second = _generate(config)
    different = _generate(
        RadiationConfig(
            source_type=RADIATION_SOURCE_DISTRIBUTED,
            placement_type=RADIATION_PLACEMENT_MIDDLE,
            seed=43,
        )
    )

    assert np.array_equal(first.field, second.field)
    assert first.metadata == second.metadata
    assert (
        first.metadata["source_position_projected"]
        != different.metadata["source_position_projected"]
    )
    assert not np.array_equal(first.field, different.field)


def test_explicit_wgs84_position_has_priority_over_polar_override():
    result = _generate(
        RadiationConfig(
            source_position=CENTER_POINT,
            source_distance_m=1_000.0,
            source_angle_deg=90.0,
        )
    )
    assert result.metadata["source_distance_m"] == pytest.approx(0.0)
    assert result.metadata["source_position_wgs84"] == pytest.approx(
        CENTER_POINT
    )


@pytest.mark.parametrize(
    ("angle_deg", "expected_delta_x", "expected_delta_y"),
    [
        (0.0, 0.0, 500.0),
        (90.0, 500.0, 0.0),
        (180.0, 0.0, -500.0),
        (270.0, -500.0, 0.0),
    ],
)
def test_source_angle_uses_clockwise_bearing_from_north(
    angle_deg, expected_delta_x, expected_delta_y
):
    result = _generate(
        RadiationConfig(
            source_distance_m=500.0,
            source_angle_deg=angle_deg,
        )
    )
    center_x, center_y = _projected_center()
    source_x, source_y = result.metadata["source_position_projected"]
    assert source_x - center_x == pytest.approx(expected_delta_x)
    assert source_y - center_y == pytest.approx(expected_delta_y)


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        (
            RadiationConfig(source_distance_m=100.0),
            ValueError,
            "provided together",
        ),
        (
            RadiationConfig(source_count=2),
            NotImplementedError,
            "exactly one source",
        ),
        (
            RadiationConfig(
                background_dose_rate=1.0,
                peak_dose_rate=1.0,
            ),
            ValueError,
            "greater than background",
        ),
        (
            RadiationConfig(measurement_model="uav"),
            NotImplementedError,
            "ground_truth",
        ),
    ],
)
def test_config_validation_errors_are_explicit(config, error_type, message):
    with pytest.raises(error_type, match=message):
        _generate(config)


def test_manual_source_outside_xlarge_extent_is_rejected():
    with pytest.raises(ValueError, match="outside the master xlarge"):
        _generate(RadiationConfig(source_position=(-1.66962, 51.42351)))


def _write_loader_dataset(
    directory: Path,
    *,
    include_optional_layers: bool,
) -> tuple[np.ndarray, np.ndarray]:
    directory.mkdir(parents=True)
    meter_per_bin = 100.0
    radius_m = 9_900.0
    center_x, center_y = _projected_center()
    shape = (198, 198)
    bounds = [
        center_x - radius_m,
        center_y - radius_m,
        center_x + radius_m,
        center_y + radius_m,
    ]
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"feature_type": "structure"},
                "geometry": {
                    "type": "Point",
                    "coordinates": list(CENTER_POINT),
                },
            }
        ],
        "environment_type": ENVIRONMENT_TYPE_FLAT,
        "climate": CLIMATE_TEMPERATE,
        "center_point": list(CENTER_POINT),
        "meter_per_bin": meter_per_bin,
        "radius_km": radius_m / 1000.0,
        "bounds": bounds,
    }
    (directory / "features.geojson").write_text(
        json.dumps(geojson_data), encoding="utf-8"
    )

    heatmap = np.ones(shape, dtype=float)
    radiation = (
        np.arange(shape[0] * shape[1], dtype=float).reshape(shape) + 0.2
    )
    np.save(directory / "heatmap.npy", heatmap)
    if include_optional_layers:
        np.save(directory / "radiation.npy", radiation)
        np.savez_compressed(
            directory / "feature_masks.npz",
            structure=np.ones(shape, dtype=np.uint8),
        )
    return heatmap, radiation


def test_loader_crops_all_optional_layers_with_same_mask(tmp_path):
    _, master_radiation = _write_loader_dataset(
        tmp_path / "extended", include_optional_layers=True
    )
    loader = DatasetLoader(str(tmp_path / "extended"))

    for size, item in loader.load_all().items():
        assert item.radiation_map is item.layers["radiation"]
        assert item.heatmap.shape == item.radiation_map.shape
        assert item.heatmap.shape == item.feature_masks["structure"].shape
        heatmap_mask = item.heatmap > 0
        assert np.array_equal(heatmap_mask, item.radiation_map > 0)
        assert np.array_equal(
            heatmap_mask, item.feature_masks["structure"] > 0
        )

        min_x_w, min_y_w, max_x_w, max_y_w = item.bounds
        master_minx, master_miny, _, _ = loader._bounds
        meter_per_bin = loader._meter_per_bin
        img_min_x = int((min_x_w - master_minx) / meter_per_bin)
        img_min_y = int((min_y_w - master_miny) / meter_per_bin)
        img_max_x = int((max_x_w - master_minx) / meter_per_bin)
        img_max_y = int((max_y_w - master_miny) / meter_per_bin)
        img_min_y, img_max_y = np.clip(
            [img_min_y, img_max_y], 0, master_radiation.shape[0]
        )
        img_min_x, img_max_x = np.clip(
            [img_min_x, img_max_x], 0, master_radiation.shape[1]
        )
        expected_crop = master_radiation[
            img_min_y:img_max_y, img_min_x:img_max_x
        ]
        assert np.array_equal(
            item.radiation_map[heatmap_mask],
            expected_crop[heatmap_mask],
        )
        assert size in {"small", "medium", "large", "xlarge"}


def test_loader_keeps_old_dataset_layers_empty(tmp_path):
    _write_loader_dataset(tmp_path / "old", include_optional_layers=False)
    item = DatasetLoader(str(tmp_path / "old")).load_environment("small")
    assert item.layers == {}
    assert item.radiation_map is None
    assert item.feature_masks == {}


class _FakeMasterEnvironment:
    def __init__(self):
        center_x, center_y = _projected_center()
        self.projected_crs = PROJECTED_CRS
        self.minx = center_x - 1_000.0
        self.miny = center_y - 1_000.0
        self.maxx = center_x + 1_000.0
        self.maxy = center_y + 1_000.0
        self.xedges = np.linspace(self.minx, self.maxx, 21)
        self.yedges = np.linspace(self.miny, self.maxy, 21)
        self._combined_heatmap = np.ones((20, 20), dtype=float)
        self.heatmaps = {
            "structure": np.ones((20, 20), dtype=np.uint8)
        }
        self.features = {
            "structure": gpd.GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            (center_x - 50.0, center_y),
                            (center_x + 50.0, center_y),
                        ]
                    )
                ],
                crs=PROJECTED_CRS,
            )
        }

    def get_combined_heatmap(self):
        return self._combined_heatmap.copy()


def test_export_dataset_writes_reusable_base_files_only(tmp_path, monkeypatch):
    generator = DataGenerator()
    fake_environment = _FakeMasterEnvironment()
    monkeypatch.setattr(
        generator,
        "generate_environment",
        lambda *args, **kwargs: fake_environment,
    )

    base_directory = tmp_path / "base"
    generator.export_dataset(
        center_point=CENTER_POINT,
        output_directory=str(base_directory),
        environment_type=ENVIRONMENT_TYPE_FLAT,
        environment_climate=CLIMATE_TEMPERATE,
        meter_per_bin=100,
    )

    assert (base_directory / "features.geojson").exists()
    assert (base_directory / "heatmap.npy").exists()
    assert (base_directory / "metadata.json").exists()
    assert not (base_directory / "radiation.npy").exists()
    assert not (base_directory / "radiation_meta.json").exists()
    assert not (base_directory / "radiation_metadata.json").exists()

    heatmap = np.load(
        base_directory / "heatmap.npy", allow_pickle=False
    )
    metadata = json.loads(
        (base_directory / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["dataset_type"] == "sarenv_base"
    assert tuple(metadata["raster_shape"]) == heatmap.shape
    assert metadata["projected_crs"] == PROJECTED_CRS
    assert metadata["raster_origin"] == "lower"
