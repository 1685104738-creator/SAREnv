"""Surface-only tests for the physical photon-fluence forward model."""

from __future__ import annotations

import numpy as np
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.surface import (
    RASTER_RESOLUTION_M,
    SurfacePhotonResponseConfig,
    SurfacePhotonResponseKernel,
    SurfaceZone,
    UniformPolygonConfig,
    UniformPolygonSource,
    ZonedPolygonConfig,
    ZonedPolygonSource,
    simulate_surface_source,
)
from shapely.geometry import box

UNIFORM_ACTIVITY_DENSITY_BQ_M2 = 1000.0
# Synthetic test surface activity density. Unit: Bq/m^2.

TEST_POLYGON_WIDTH_M = 10.0
TEST_POLYGON_HEIGHT_M = 10.0
EXPECTED_SOURCE_CELL_COUNT = 100

ZONE_1_ACTIVITY_DENSITY_BQ_M2 = 1000.0
ZONE_2_ACTIVITY_DENSITY_BQ_M2 = 5000.0

TEST_GRID_NX = 20
TEST_GRID_NY = 20

LINEAR_SCALE_FACTOR = 2.0
NORMALIZATION_CHECK_SCALE_FACTOR = 10.0

FFT_DIRECT_RTOL = 1e-10
FFT_DIRECT_ATOL = 1e-12
LINEAR_SCALING_RTOL = 1e-10
LINEAR_SCALING_ATOL = 1e-12
SYMMETRY_RTOL = 1e-9
SYMMETRY_ATOL = 1e-12
SOURCE_CONSERVATION_RTOL = 1e-12
SOURCE_CONSERVATION_ATOL = 1e-9

# Existing projected-metre CRS used by the project's radiation tests.
TEST_CRS = "EPSG:32630"


def _grid() -> GridSpec:
    return GridSpec.from_bounds(
        (0.0, 0.0, TEST_GRID_NX, TEST_GRID_NY),
        TEST_CRS,
        resolution_m=RASTER_RESOLUTION_M,
    )


def _polygon_bounds() -> tuple[float, float, float, float]:
    minx = (TEST_GRID_NX * RASTER_RESOLUTION_M - TEST_POLYGON_WIDTH_M) / 2.0
    miny = (TEST_GRID_NY * RASTER_RESOLUTION_M - TEST_POLYGON_HEIGHT_M) / 2.0
    return (
        minx,
        miny,
        minx + TEST_POLYGON_WIDTH_M,
        miny + TEST_POLYGON_HEIGHT_M,
    )


def _source(activity_density_bq_m2: float) -> UniformPolygonSource:
    return UniformPolygonSource(
        UniformPolygonConfig(
            source_id="physical-uniform-test",
            geometry=box(*_polygon_bounds()),
            activity_density_bq_m2=activity_density_bq_m2,
            crs=TEST_CRS,
        )
    )


def _response_config() -> SurfacePhotonResponseConfig:
    return SurfacePhotonResponseConfig()


def _simulate(activity_density_bq_m2: float):
    grid = _grid()
    kernel = SurfacePhotonResponseKernel.create(_response_config(), grid)
    return simulate_surface_source(
        _source(activity_density_bq_m2),
        kernel,
        nominal_grid=grid,
    )


def _direct_superposition(
    cell_activity_bq: np.ndarray,
    grid: GridSpec,
    config: SurfacePhotonResponseConfig,
) -> np.ndarray:
    """Brute-force test reference; this is not part of the simulation path."""
    result = np.zeros(grid.shape, dtype=np.float64)
    for observation_row in range(grid.height):
        for observation_column in range(grid.width):
            total = np.float64(0.0)
            for source_row in range(grid.height):
                for source_column in range(grid.width):
                    dx_m = (
                        observation_column - source_column
                    ) * grid.resolution_m
                    dy_m = (observation_row - source_row) * grid.resolution_m
                    distance_m = np.sqrt(
                        dx_m**2
                        + dy_m**2
                        + config.observation_height_m**2
                    )
                    response_per_bq = (
                        config.gamma_yield_per_decay
                        * np.exp(-config.mu_air_m_inv * distance_m)
                        / (4.0 * np.pi * distance_m**2)
                    )
                    total += (
                        cell_activity_bq[source_row, source_column]
                        * response_per_bq
                    )
            result[observation_row, observation_column] = total
    return result


def test_source_activity_conservation() -> None:
    result = _simulate(UNIFORM_ACTIVITY_DENSITY_BQ_M2)
    source_cell_count = int(np.count_nonzero(result.activity_density_bq_m2))
    expected_total_activity_bq = (
        UNIFORM_ACTIVITY_DENSITY_BQ_M2
        * TEST_POLYGON_WIDTH_M
        * TEST_POLYGON_HEIGHT_M
    )
    raster_total_activity_bq = float(result.cell_activity_bq.sum())
    relative_error = abs(
        raster_total_activity_bq - expected_total_activity_bq
    ) / expected_total_activity_bq

    assert source_cell_count == EXPECTED_SOURCE_CELL_COUNT
    assert result.activity_density_bq_m2.dtype == np.float64
    assert result.cell_activity_bq.dtype == np.float64
    assert result.kernel.values.dtype == np.float64
    assert (
        result.photon_fluence_rate_patch.photon_fluence_rate.dtype
        == np.float64
    )
    assert np.isclose(
        raster_total_activity_bq,
        expected_total_activity_bq,
        rtol=SOURCE_CONSERVATION_RTOL,
        atol=SOURCE_CONSERVATION_ATOL,
    )
    print("=== Surface Radiation Physical Model Test ===")
    print(f"Raster resolution: {RASTER_RESOLUTION_M} m")
    print(
        "Polygon area: "
        f"{TEST_POLYGON_WIDTH_M * TEST_POLYGON_HEIGHT_M} m^2"
    )
    print(
        "Activity density: "
        f"{UNIFORM_ACTIVITY_DENSITY_BQ_M2} Bq/m^2"
    )
    print(f"Expected total activity: {expected_total_activity_bq} Bq")
    print(f"Raster total activity: {raster_total_activity_bq} Bq")
    print(f"Relative source conservation error: {relative_error * 100.0} %")


def test_linear_scaling() -> None:
    base = _simulate(UNIFORM_ACTIVITY_DENSITY_BQ_M2)
    scaled = _simulate(
        UNIFORM_ACTIVITY_DENSITY_BQ_M2 * LINEAR_SCALE_FACTOR
    )
    base_field = base.photon_fluence_rate_patch.photon_fluence_rate
    scaled_field = scaled.photon_fluence_rate_patch.photon_fluence_rate
    expected_scaled_field = LINEAR_SCALE_FACTOR * base_field
    max_abs_error = float(np.max(np.abs(scaled_field - expected_scaled_field)))
    assert np.allclose(
        scaled_field,
        expected_scaled_field,
        rtol=LINEAR_SCALING_RTOL,
        atol=LINEAR_SCALING_ATOL,
    )
    print(f"Linear scaling max abs error: {max_abs_error}")
    print("Linear scaling test: PASS")


def test_symmetry() -> None:
    field = _simulate(
        UNIFORM_ACTIVITY_DENSITY_BQ_M2
    ).photon_fluence_rate_patch.photon_fluence_rate
    assert np.allclose(
        field,
        np.flipud(field),
        rtol=SYMMETRY_RTOL,
        atol=SYMMETRY_ATOL,
    )
    assert np.allclose(
        field,
        np.fliplr(field),
        rtol=SYMMETRY_RTOL,
        atol=SYMMETRY_ATOL,
    )
    symmetry_max_abs_error = float(
        max(
            np.max(np.abs(field - np.flipud(field))),
            np.max(np.abs(field - np.fliplr(field))),
        )
    )
    print(f"Symmetry max abs error: {symmetry_max_abs_error}")
    print("Symmetry test: PASS")


def test_spatial_behaviour_without_cutoff_or_circular_wraparound() -> None:
    field = _simulate(
        UNIFORM_ACTIVITY_DENSITY_BQ_M2
    ).photon_fluence_rate_patch.photon_fluence_rate
    center_row = TEST_GRID_NY // 2
    source_min_column = int(
        (_polygon_bounds()[0]) / RASTER_RESOLUTION_M
    )
    outside_profile = field[center_row, :source_min_column]
    first_source_value = field[center_row, source_min_column]

    assert np.all(outside_profile > 0.0)
    assert np.all(np.diff(outside_profile) > 0.0)
    assert first_source_value > outside_profile[-1]
    print(
        "Spatial profile (far, near outside, first source cell): "
        f"{outside_profile[0]}, {outside_profile[-1]}, {first_source_value}"
    )
    print("Spatial decay / no hard cutoff test: PASS")


def test_no_normalisation() -> None:
    base_field = _simulate(
        UNIFORM_ACTIVITY_DENSITY_BQ_M2
    ).photon_fluence_rate_patch.photon_fluence_rate
    scaled_activity_density = (
        UNIFORM_ACTIVITY_DENSITY_BQ_M2
        * NORMALIZATION_CHECK_SCALE_FACTOR
    )
    scaled_field = _simulate(
        scaled_activity_density
    ).photon_fluence_rate_patch.photon_fluence_rate

    assert np.isclose(
        scaled_field.max(),
        NORMALIZATION_CHECK_SCALE_FACTOR * base_field.max(),
        rtol=LINEAR_SCALING_RTOL,
        atol=LINEAR_SCALING_ATOL,
    )
    maximum_field_ratio = float(scaled_field.max() / base_field.max())
    print(f"Input activity density: {UNIFORM_ACTIVITY_DENSITY_BQ_M2} Bq/m^2")
    print(f"Photon fluence field min: {base_field.min()} photons/m^2/s")
    print(f"Photon fluence field max: {base_field.max()} photons/m^2/s")
    print(f"Scaled activity density: {scaled_activity_density} Bq/m^2")
    print(f"Scaled field max: {scaled_field.max()} photons/m^2/s")
    print(f"Maximum field ratio: {maximum_field_ratio}")
    print("Normalization detected: NO")


def test_fft_matches_direct_superposition() -> None:
    result = _simulate(UNIFORM_ACTIVITY_DENSITY_BQ_M2)
    fft_field = result.photon_fluence_rate_patch.photon_fluence_rate
    direct_field = _direct_superposition(
        result.cell_activity_bq,
        result.grid,
        result.kernel.config,
    )
    max_abs_error = float(np.max(np.abs(fft_field - direct_field)))
    assert np.allclose(
        fft_field,
        direct_field,
        rtol=FFT_DIRECT_RTOL,
        atol=FFT_DIRECT_ATOL,
    )
    print(f"FFT vs direct max abs error: {max_abs_error}")
    print("FFT vs direct: PASS")


def test_zoned_polygon_uses_activity_density_and_last_defined_wins() -> None:
    grid = _grid()
    minx, miny, maxx, maxy = _polygon_bounds()
    split_x = minx + TEST_POLYGON_WIDTH_M / 2.0
    zones = (
        SurfaceZone(
            "zone-1",
            box(minx, miny, maxx, maxy),
            ZONE_1_ACTIVITY_DENSITY_BQ_M2,
        ),
        SurfaceZone(
            "zone-2",
            box(split_x, miny, maxx, maxy),
            ZONE_2_ACTIVITY_DENSITY_BQ_M2,
        ),
    )
    source = ZonedPolygonSource(
        ZonedPolygonConfig("physical-zoned-test", zones, TEST_CRS)
    )
    activity_density = source.rasterize(grid)
    source_min_column = int(minx / RASTER_RESOLUTION_M)
    split_column = int(split_x / RASTER_RESOLUTION_M)
    source_row = int(miny / RASTER_RESOLUTION_M)

    assert activity_density[source_row, source_min_column] == (
        ZONE_1_ACTIVITY_DENSITY_BQ_M2
    )
    assert activity_density[source_row, split_column] == (
        ZONE_2_ACTIVITY_DENSITY_BQ_M2
    )
    assert activity_density.dtype == np.float64
