"""Tests for offline radiation generation on reusable base datasets."""

import hashlib
import json
from pathlib import Path

import numpy as np
import osmnx as ox
from pyproj import Transformer

from sarenv import RadiationConfig, generate_radiation_layer
from sarenv.core import generation
from sarenv.io import osm_query
from sarenv.utils.geo import get_utm_epsg


CENTER_POINT = (-2.66962, 51.42351)
ENVIRONMENT_TYPE = "flat"
CLIMATE = "temperate"
METER_PER_BIN = 100.0
RASTER_SHAPE = (40, 40)


def _file_fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, path.stat().st_mtime_ns


def _write_base_dataset(
    directory: Path,
    *,
    include_metadata_json: bool = True,
) -> dict:
    directory.mkdir(parents=True)
    projected_crs = get_utm_epsg(*CENTER_POINT)
    transformer = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )
    center_x, center_y = transformer.transform(*CENTER_POINT)
    half_width_m = RASTER_SHAPE[1] * METER_PER_BIN / 2.0
    bounds = [
        center_x - half_width_m,
        center_y - half_width_m,
        center_x + half_width_m,
        center_y + half_width_m,
    ]
    metadata = {
        "schema_version": 1,
        "dataset_type": "sarenv_base",
        "center_point": list(CENTER_POINT),
        "environment_type": ENVIRONMENT_TYPE,
        "climate": CLIMATE,
        "meter_per_bin": METER_PER_BIN,
        "radius_km": 9.9,
        "projected_crs": projected_crs,
        "bounds_projected": bounds,
        "raster_shape": list(RASTER_SHAPE),
        "raster_origin": "lower",
        "raster_axis_order": "row_y_column_x",
    }
    features = {
        "type": "FeatureCollection",
        "features": [],
        "center_point": list(CENTER_POINT),
        "environment_type": ENVIRONMENT_TYPE,
        "climate": CLIMATE,
        "meter_per_bin": METER_PER_BIN,
        "radius_km": 9.9,
        "bounds": bounds,
    }
    (directory / "features.geojson").write_text(
        json.dumps(features), encoding="utf-8"
    )
    np.save(
        directory / "heatmap.npy",
        np.full(RASTER_SHAPE, 1.0 / np.prod(RASTER_SHAPE)),
    )
    if include_metadata_json:
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
    return metadata


def _forbid_network_and_environment_generation(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError(
            "Offline radiation generation attempted OSM/environment access."
        )

    monkeypatch.setattr(osm_query, "query_features", unexpected_call)
    monkeypatch.setattr(ox, "features_from_polygon", unexpected_call)
    monkeypatch.setattr(
        generation.Environment,
        "_load_features",
        unexpected_call,
    )


def test_offline_generation_preserves_base_files_and_alignment(
    tmp_path,
    monkeypatch,
):
    dataset_directory = tmp_path / "base"
    base_metadata = _write_base_dataset(dataset_directory)
    base_paths = [
        dataset_directory / "features.geojson",
        dataset_directory / "heatmap.npy",
        dataset_directory / "metadata.json",
    ]
    before = {path.name: _file_fingerprint(path) for path in base_paths}
    _forbid_network_and_environment_generation(monkeypatch)

    result = generate_radiation_layer(
        dataset_directory,
        RadiationConfig(
            source_type="distributed",
            shape_type="stretched",
            seed=42,
        ),
    )

    after = {path.name: _file_fingerprint(path) for path in base_paths}
    heatmap = np.load(
        dataset_directory / "heatmap.npy", allow_pickle=False
    )
    radiation = np.load(
        dataset_directory / "radiation.npy", allow_pickle=False
    )
    radiation_metadata = json.loads(
        (dataset_directory / "radiation_meta.json").read_text(
            encoding="utf-8"
        )
    )

    assert before == after
    assert radiation.shape == heatmap.shape == RASTER_SHAPE
    assert np.array_equal(result.field, radiation)
    assert radiation_metadata["projected_crs"] == base_metadata["projected_crs"]
    assert radiation_metadata["meter_per_bin"] == METER_PER_BIN
    assert (
        radiation_metadata["raster_bounds_projected"]
        == base_metadata["bounds_projected"]
    )
    assert radiation_metadata["raster_shape"] == list(RASTER_SHAPE)
    assert radiation_metadata["raster_origin"] == "lower"
    assert radiation_metadata["raster_axis_order"] == "row_y_column_x"
    assert radiation_metadata["base_metadata_source"] == "metadata.json"
    assert radiation_metadata["base_radius_km"] == 9.9


def test_changed_config_overwrites_only_radiation_outputs(
    tmp_path,
    monkeypatch,
):
    dataset_directory = tmp_path / "base"
    _write_base_dataset(dataset_directory)
    base_paths = [
        dataset_directory / "features.geojson",
        dataset_directory / "heatmap.npy",
        dataset_directory / "metadata.json",
    ]
    before_base = {
        path.name: _file_fingerprint(path) for path in base_paths
    }
    _forbid_network_and_environment_generation(monkeypatch)

    generate_radiation_layer(
        dataset_directory,
        RadiationConfig(
            source_type="distributed",
            shape_type="circular",
            seed=42,
        ),
    )
    first_radiation = hashlib.sha256(
        (dataset_directory / "radiation.npy").read_bytes()
    ).hexdigest()
    first_metadata = hashlib.sha256(
        (dataset_directory / "radiation_meta.json").read_bytes()
    ).hexdigest()

    generate_radiation_layer(
        dataset_directory,
        RadiationConfig(
            source_type="point",
            intensity_type="high",
            seed=99,
        ),
    )
    second_radiation = hashlib.sha256(
        (dataset_directory / "radiation.npy").read_bytes()
    ).hexdigest()
    second_metadata = hashlib.sha256(
        (dataset_directory / "radiation_meta.json").read_bytes()
    ).hexdigest()
    after_base = {
        path.name: _file_fingerprint(path) for path in base_paths
    }

    assert before_base == after_base
    assert first_radiation != second_radiation
    assert first_metadata != second_metadata


def test_legacy_dataset_metadata_fallback_is_read_only(tmp_path):
    dataset_directory = tmp_path / "legacy"
    _write_base_dataset(
        dataset_directory,
        include_metadata_json=False,
    )
    base_paths = [
        dataset_directory / "features.geojson",
        dataset_directory / "heatmap.npy",
    ]
    before = {path.name: _file_fingerprint(path) for path in base_paths}

    result = generate_radiation_layer(
        dataset_directory,
        RadiationConfig(seed=7),
    )

    after = {path.name: _file_fingerprint(path) for path in base_paths}
    assert before == after
    assert result.metadata["base_metadata_source"] == "features.geojson"
    assert not (dataset_directory / "metadata.json").exists()


def test_named_scenario_does_not_replace_default_layer(tmp_path):
    dataset_directory = tmp_path / "base"
    _write_base_dataset(dataset_directory)
    generate_radiation_layer(
        dataset_directory,
        RadiationConfig(seed=42),
    )
    default_fingerprint = _file_fingerprint(
        dataset_directory / "radiation.npy"
    )

    generate_radiation_layer(
        dataset_directory,
        RadiationConfig(seed=43),
        scenario_name="radiation_scenario_b",
    )

    assert _file_fingerprint(
        dataset_directory / "radiation.npy"
    ) == default_fingerprint
    assert (dataset_directory / "radiation_scenario_b.npy").exists()
    assert (dataset_directory / "radiation_scenario_b_meta.json").exists()
