"""Separation and compatibility tests for the radiation package."""

from __future__ import annotations

import json
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from sarenv import DataGenerator
from sarenv.core.radiation import RadiationConfig, generate_radiation_field
from sarenv.io.radiation_layer import generate_radiation_layer
from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    GridSpec,
    TerrainContext,
    UniformPolygonConfig,
    UniformPolygonSource,
    simulate_surface_source,
)


def _write_terrain_dataset(directory):
    directory.mkdir()
    center = (-1.5486, 51.4214)
    features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"feature_type": "road"},
                "geometry": {"type": "Point", "coordinates": list(center)},
            }
        ],
        "center_point": list(center),
        "bounds": [450_000.0, 5_690_000.0, 470_000.0, 5_710_000.0],
    }
    metadata = {
        "center_point": list(center),
        "projected_crs": "EPSG:32630",
        "bounds_projected": features["bounds"],
    }
    (directory / "features.geojson").write_text(json.dumps(features), encoding="utf-8")
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_terrain_context_uses_shared_geography_without_heatmap_or_network(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "base"
    _write_terrain_dataset(dataset)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("TerrainContext attempted an OSM network query.")

    monkeypatch.setattr(
        "sarenv.io.osm_query.query_features", network_forbidden
    )
    context = TerrainContext.from_dataset(dataset)
    assert context.projected_crs == "EPSG:32630"
    assert len(context.features) == 1
    assert not (dataset / "heatmap.npy").exists()
    assert context.contains_geometry(box(455_000, 5_695_000, 456_000, 5_696_000))
    context.validate_grid(
        GridSpec.from_bounds((455_000, 5_695_000, 455_010, 5_695_010), "EPSG:32630")
    )
    with pytest.raises(ValueError, match="outside"):
        context.validate_grid(
            GridSpec.from_bounds((449_990, 5_695_000, 450_010, 5_695_010), "EPSG:32630")
        )


def test_surface_generation_does_not_read_sar_heatmap(tmp_path):
    dataset = tmp_path / "base"
    _write_terrain_dataset(dataset)
    context = TerrainContext.from_dataset(dataset)
    source = UniformPolygonSource(
        UniformPolygonConfig(
            "surface",
            box(455_000, 5_695_000, 455_010, 5_695_008),
            5.0,
            context.projected_crs,
        )
    )
    kernel = BenchmarkSurfaceResponseKernel.create(
        BenchmarkSurfaceResponseConfig(1.0, 3.0)
    )
    result = simulate_surface_source(source, kernel)
    assert result.nominal_surface_field.shape == (8, 10)
    assert result.dose_rate_patch.excess_uSv_h.shape == (14, 16)
    assert result.nominal_grid.resolution_m == 1.0


def test_legacy_heatmap_aligned_api_is_deprecated_and_disabled():
    with pytest.warns(DeprecationWarning, match="shared RadiationConfig"):
        legacy_config = RadiationConfig()
    with pytest.warns(DeprecationWarning, match="shared RadiationConfig"):
        with pytest.raises(NotImplementedError, match="independent 1 m"):
            generate_radiation_field(legacy_config)
    with pytest.warns(DeprecationWarning, match="heatmap.npy"):
        with pytest.raises(NotImplementedError, match="independent 1 m"):
            generate_radiation_layer("unused", legacy_config)


def test_base_sar_import_does_not_import_radiation_package():
    script = (
        "import sys, sarenv; "
        "loaded=[n for n in sys.modules if n.startswith('sarenv.radiation') "
        "or n in {'sarenv.core.radiation','sarenv.io.radiation_layer'}]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


class _FakeMasterEnvironment:
    projected_crs = "EPSG:32630"
    minx = 0.0
    miny = 0.0
    maxx = 200.0
    maxy = 200.0
    xedges = np.linspace(0.0, 200.0, 3)
    yedges = np.linspace(0.0, 200.0, 3)
    heatmaps = {"structure": np.ones((2, 2), dtype=np.uint8)}
    features = {
        "structure": gpd.GeoDataFrame(
            geometry=[LineString([(10.0, 10.0), (20.0, 20.0)])],
            crs="EPSG:32630",
        )
    }

    def get_combined_heatmap(self):
        return np.ones((2, 2), dtype=float)


def test_base_data_generator_remains_radiation_free(tmp_path, monkeypatch):
    generator = DataGenerator()
    monkeypatch.setattr(
        generator,
        "generate_environment",
        lambda *args, **kwargs: _FakeMasterEnvironment(),
    )
    output = tmp_path / "base"
    generator.export_dataset(
        center_point=(-1.5486, 51.4214),
        output_directory=str(output),
        environment_type="flat",
        environment_climate="temperate",
        meter_per_bin=100,
    )
    assert (output / "features.geojson").exists()
    assert (output / "heatmap.npy").exists()
    assert (output / "metadata.json").exists()
    assert not (output / "radiation.npy").exists()
    assert not (output / "radiation_meta.json").exists()
