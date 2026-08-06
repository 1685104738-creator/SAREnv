"""Numerical tests for independent deterministic radiation models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from sarenv import DatasetLoader
from sarenv.radiation import (
    BenchmarkSurfaceResponseConfig,
    BenchmarkSurfaceResponseKernel,
    CompositeRadiationField,
    DoseRatePatch,
    GridSpec,
    PointSource,
    PointSourceConfig,
    SurfaceZone,
    UniformPolygonConfig,
    UniformPolygonSource,
    ZonedPolygonConfig,
    ZonedPolygonSource,
    load_point_source,
    load_surface_simulation,
    save_point_source,
    save_surface_simulation,
    simulate_surface_source,
)


CRS = "EPSG:32630"


def _kernel(cutoff: float = 2.0):
    return BenchmarkSurfaceResponseKernel.create(
        BenchmarkSurfaceResponseConfig(
            core_radius_m=1.0,
            cutoff_radius_m=cutoff,
        )
    )


def _uniform(
    geometry,
    intensity: float,
    source_id: str = "uniform",
):
    return UniformPolygonSource(
        UniformPolygonConfig(
            source_id=source_id,
            geometry=geometry,
            nominal_surface_field_uSv_h=intensity,
            crs=CRS,
        )
    )


def test_grid_spec_coordinate_contract_and_boundaries():
    grid = GridSpec.from_bounds((0.2, 0.3, 3.7, 2.2), CRS)
    assert grid.bounds == (0.0, 0.0, 4.0, 3.0)
    assert grid.shape == (3, 4)
    assert grid.row_col_to_world(0, 0) == (0.5, 0.5)
    assert grid.row_col_to_world(2, 3) == (3.5, 2.5)
    assert grid.world_to_row_col(0.5, 0.5) == (0, 0)
    assert grid.world_to_row_col(3.999, 2.999) == (2, 3)
    assert grid.contains(0.0, 0.0)
    assert not grid.contains(4.0, 2.0)
    assert not grid.contains(2.0, 3.0)
    with pytest.raises(ValueError, match="outside"):
        grid.world_to_row_col(4.0, 2.0)
    assert GridSpec.from_metadata(grid.to_metadata()) == grid


def test_grid_resolution_is_independent_and_fixed_to_one_metre():
    with pytest.raises(ValueError, match="fixed at 1.0 m"):
        GridSpec.from_bounds((0, 0, 10, 10), CRS, resolution_m=30.0)
    with pytest.raises(ValueError, match="projected CRS"):
        GridSpec.from_bounds((0, 0, 10, 10), "EPSG:4326")


def test_uniform_polygon_rasterisation_inside_outside_and_patch_edge():
    grid = GridSpec.from_bounds((0, 0, 5, 5), CRS)
    source = _uniform(box(1, 1, 4, 4), 7.0)
    field = source.rasterize(grid)
    expected = np.zeros((5, 5))
    expected[1:4, 1:4] = 7.0
    assert np.array_equal(field, expected)

    edge_source = _uniform(box(0, 0, 1, 1), 3.0)
    edge_field = edge_source.rasterize(grid)
    assert edge_field[0, 0] == 3.0
    assert np.count_nonzero(edge_field) == 1


def test_irregular_polygon_rasterisation_uses_cell_centres():
    grid = GridSpec.from_bounds((0, 0, 5, 5), CRS)
    triangle = Polygon([(0.1, 0.1), (4.8, 0.2), (0.2, 4.7)])
    field = _uniform(triangle, 4.0).rasterize(grid)
    assert field[0, 0] == 4.0
    assert field[4, 4] == 0.0
    assert set(np.unique(field)) == {0.0, 4.0}


def test_zoned_polygon_last_defined_wins_and_order_is_repeatable():
    first = SurfaceZone("base", box(0, 0, 4, 4), 1.0)
    second = SurfaceZone("inner", box(2, 2, 5, 5), 5.0)
    grid = GridSpec.from_bounds((0, 0, 5, 5), CRS)
    config = ZonedPolygonConfig("zones", (first, second), CRS)
    source = ZonedPolygonSource(config)
    field_a = source.rasterize(grid)
    field_b = source.rasterize(grid)
    assert np.array_equal(field_a, field_b)
    assert field_a[1, 1] == 1.0
    assert field_a[2, 2] == 5.0
    assert source.source_metadata()["overlap_rule"] == "last_defined_wins"

    reverse = ZonedPolygonSource(
        ZonedPolygonConfig("reverse", (second, first), CRS)
    ).rasterize(grid)
    assert reverse[2, 2] == 1.0


def test_benchmark_kernel_shape_cutoff_normalisation_and_symmetry():
    kernel = _kernel(cutoff=3.0)
    assert kernel.values.shape == (7, 7)
    assert kernel.center_index == (3, 3)
    assert np.all(kernel.values >= 0)
    assert kernel.values[0, 0] == 0.0
    assert kernel.values[3, 3] == kernel.values.max()
    assert np.isclose(kernel.values.sum(), 1.0, atol=1e-12)
    assert np.array_equal(kernel.values, np.flipud(kernel.values))
    assert np.array_equal(kernel.values, np.fliplr(kernel.values))


def test_single_pixel_full_convolution_equals_scaled_kernel():
    grid = GridSpec.from_bounds((0, 0, 3, 3), CRS)
    result = simulate_surface_source(
        _uniform(box(1, 1, 2, 2), 6.0),
        _kernel(cutoff=2.0),
        nominal_grid=grid,
    )
    expected = np.zeros((7, 7))
    expected[1:6, 1:6] = 6.0 * result.kernel.values
    assert np.allclose(result.dose_rate_patch.excess_uSv_h, expected, atol=1e-12)


def test_large_uniform_area_preserves_centre_and_has_external_tail():
    source = _uniform(box(0, 0, 21, 21), 10.0)
    result = simulate_surface_source(source, _kernel(cutoff=3.0))
    patch = result.dose_rate_patch
    assert patch.query_excess_dose_rate(10.5, 10.5, method="nearest") == pytest.approx(10.0)
    edge = patch.query_excess_dose_rate(0.5, 10.5, method="nearest")
    tail = patch.query_excess_dose_rate(-0.5, 10.5, method="nearest")
    assert 0.0 < tail < edge < 10.0
    assert patch.query_excess_dose_rate(-3.1, 10.5) == 0.0


def test_surface_simulation_linearity_and_intensity_scaling():
    grid = GridSpec.from_bounds((0, 0, 5, 5), CRS)
    kernel = _kernel()
    result_a = simulate_surface_source(
        _uniform(box(0, 0, 5, 5), 2.0, "a"), kernel, nominal_grid=grid
    )
    result_b = simulate_surface_source(
        _uniform(box(0, 0, 5, 5), 3.0, "b"), kernel, nominal_grid=grid
    )
    result_sum = simulate_surface_source(
        _uniform(box(0, 0, 5, 5), 5.0, "sum"), kernel, nominal_grid=grid
    )
    result_double = simulate_surface_source(
        _uniform(box(0, 0, 5, 5), 4.0, "double"), kernel, nominal_grid=grid
    )
    assert np.allclose(
        result_sum.dose_rate_patch.excess_uSv_h,
        result_a.dose_rate_patch.excess_uSv_h
        + result_b.dose_rate_patch.excess_uSv_h,
    )
    assert np.allclose(
        result_double.dose_rate_patch.excess_uSv_h,
        2.0 * result_a.dose_rate_patch.excess_uSv_h,
    )


def _direct_full_convolution(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    output = np.zeros(
        (
            field.shape[0] + kernel.shape[0] - 1,
            field.shape[1] + kernel.shape[1] - 1,
        )
    )
    for row in range(field.shape[0]):
        for col in range(field.shape[1]):
            output[
                row : row + kernel.shape[0],
                col : col + kernel.shape[1],
            ] += field[row, col] * kernel
    return output


def test_fft_convolution_matches_direct_reference_sum():
    zones = (
        SurfaceZone("a", box(0, 0, 1, 1), 1.0),
        SurfaceZone("b", box(1, 0, 2, 1), 2.0),
        SurfaceZone("c", box(0, 1, 1, 2), 3.0),
        SurfaceZone("d", box(1, 1, 2, 2), 4.0),
    )
    source = ZonedPolygonSource(ZonedPolygonConfig("small", zones, CRS))
    result = simulate_surface_source(
        source,
        _kernel(cutoff=1.0),
        nominal_grid=GridSpec.from_bounds((0, 0, 2, 2), CRS),
    )
    direct = _direct_full_convolution(
        result.nominal_surface_field, result.kernel.values
    )
    assert np.allclose(result.dose_rate_patch.excess_uSv_h, direct, atol=1e-12)


def test_full_convolution_shape_bounds_and_cell_alignment():
    input_grid = GridSpec.from_bounds((10, 20, 14, 23), CRS)
    result = simulate_surface_source(
        _uniform(box(10, 20, 14, 23), 2.0),
        _kernel(cutoff=2.0),
        nominal_grid=input_grid,
    )
    patch = result.dose_rate_patch
    assert patch.excess_uSv_h.shape == (7, 8)
    assert patch.grid.bounds == (8.0, 18.0, 16.0, 25.0)
    assert patch.grid.row_col_to_world(2, 2) == input_grid.row_col_to_world(0, 0)
    assert result.dose_rate_metadata["convolution_mode"] == "full"


def test_point_source_is_analytic_excess_only_and_deterministic():
    source = PointSource(
        PointSourceConfig(
            source_id="point-a",
            x_m=100.0,
            y_m=200.0,
            reference_excess_uSv_h=50.0,
            crs=CRS,
            cutoff_radius_m=10.0,
        )
    )
    assert source.query_excess_dose_rate(100.0, 200.0) == pytest.approx(50.0)
    near = source.query_excess_dose_rate(101.0, 200.0)
    far = source.query_excess_dose_rate(104.0, 200.0)
    assert 0.0 < far < near < 50.0
    assert source.query_excess_dose_rate(120.0, 200.0) == 0.0
    grid = GridSpec.from_bounds((98, 198, 103, 203), CRS)
    assert np.array_equal(source.rasterize(grid), source.rasterize(grid))
    assert source.to_metadata()["background_included"] is False


def test_composite_adds_sources_and_background_exactly_once():
    grid = GridSpec.from_bounds((0, 0, 1, 1), CRS)
    patch_a = DoseRatePatch(np.array([[2.0]]), grid)
    patch_b = DoseRatePatch(np.array([[3.0]]), grid)
    point = PointSource(
        PointSourceConfig(
            source_id="point",
            x_m=0.5,
            y_m=0.5,
            reference_excess_uSv_h=10.0,
            crs=CRS,
        )
    )
    field = CompositeRadiationField(
        crs=CRS,
        background_uSv_h=0.2,
        point_sources=(point,),
        surface_patches=(patch_a, patch_b),
    )
    assert field.query_excess_dose_rate(0.5, 0.5) == pytest.approx(15.0)
    assert field.query_total_dose_rate(0.5, 0.5) == pytest.approx(15.2)
    assert field.query_total_dose_rate(50.0, 50.0) > 0.2

    surface_only = CompositeRadiationField(
        CRS, 0.2, surface_patches=(patch_a, patch_b)
    )
    assert surface_only.query_total_dose_rate(0.5, 0.5) == pytest.approx(5.2)
    assert surface_only.query_total_dose_rate(2.0, 2.0) == pytest.approx(0.2)
    assert surface_only.query_total_dose_rate(1.0, 0.5) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="NaN or infinite"):
        DoseRatePatch(np.array([[np.nan]]), grid)


def test_surface_simulation_and_io_are_fully_deterministic(tmp_path, monkeypatch):
    def fail_rng(*args, **kwargs):
        raise AssertionError("Radiation generation must not use random numbers.")

    monkeypatch.setattr(np.random, "default_rng", fail_rng)
    source = _uniform(box(0, 0, 6, 4), 8.0)
    first = simulate_surface_source(source, _kernel())
    second = simulate_surface_source(source, _kernel())
    assert np.array_equal(first.nominal_surface_field, second.nominal_surface_field)
    assert np.array_equal(
        first.dose_rate_patch.excess_uSv_h,
        second.dose_rate_patch.excess_uSv_h,
    )

    paths = save_surface_simulation(first, tmp_path / "surface")
    assert paths.nominal_field.name == "nominal_surface_field.npy"
    assert paths.dose_rate_patch.name == "dose_rate_patch.npy"
    loaded = load_surface_simulation(tmp_path / "surface")
    assert np.array_equal(loaded.nominal_surface_field, first.nominal_surface_field)
    assert np.array_equal(
        loaded.dose_rate_patch.excess_uSv_h,
        first.dose_rate_patch.excess_uSv_h,
    )
    assert loaded.dose_rate_metadata["background_included"] is False
    assert loaded.dose_rate_metadata["data_content"] == "source_excess_only"


def test_point_source_metadata_round_trip(tmp_path):
    source = PointSource(PointSourceConfig("point", 1.0, 2.0, 3.0, CRS))
    path = save_point_source(source, tmp_path / "point_source_meta.json")
    restored = load_point_source(path)
    assert restored == source
    assert restored.query_excess_dose_rate(1.0, 2.0) == 3.0


def _write_sar_loader_fixture(directory: Path, *, radiation_shape=(3, 4)):
    directory.mkdir()
    center = (-1.5486, 51.4214)
    bounds = [450_000.0, 5_690_000.0, 469_800.0, 5_709_800.0]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"feature_type": "structure"},
                "geometry": {"type": "Point", "coordinates": list(center)},
            }
        ],
        "environment_type": "flat",
        "climate": "temperate",
        "center_point": list(center),
        "meter_per_bin": 100.0,
        "radius_km": 9.9,
        "bounds": bounds,
    }
    (directory / "features.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    shape = (198, 198)
    np.save(directory / "heatmap.npy", np.ones(shape))
    np.save(directory / "radiation.npy", np.ones(radiation_shape))
    np.savez_compressed(directory / "feature_masks.npz", structure=np.ones(shape))


def test_sar_loader_ignores_independent_mismatched_radiation_patch(tmp_path):
    dataset = tmp_path / "dataset"
    _write_sar_loader_fixture(dataset)
    item = DatasetLoader(str(dataset)).load_environment("small")
    assert item is not None
    assert "radiation" not in item.layers
    assert item.heatmap.shape == item.feature_masks["structure"].shape
    with pytest.warns(DeprecationWarning, match="independent radiation"):
        assert item.radiation_map is None
