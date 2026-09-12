"""Surface-only Cs-137 photon-fluence and collision-air-kerma tests."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

import sarenv.radiation.surface.simulator as surface_simulator
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.surface import (
    AIR_DENSITY_KG_M3,
    COLLISION_AIR_KERMA_RATE_UNIT,
    CS137_FLUENCE_TO_KERMA_UGY_H,
    GAMMA_ENERGY_MEV,
    GAMMA_YIELD_PER_DECAY,
    GY_TO_UGY,
    HALF_LIFE_YEARS,
    HIGH_ACTIVITY_DENSITY_BQ_M2,
    LOW_ACTIVITY_DENSITY_BQ_M2,
    MASS_ATTENUATION_COEFF_AIR_M2_KG,
    MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG,
    MEDIUM_ACTIVITY_DENSITY_BQ_M2,
    MEV_TO_J,
    MU_AIR_M_INV,
    OBSERVATION_HEIGHT_M,
    RADIONUCLIDE_NAME,
    RASTER_RESOLUTION_M,
    SECONDS_PER_HOUR,
    SurfacePhotonResponseConfig,
    SurfacePhotonResponseKernel,
    UniformPolygonConfig,
    UniformPolygonSource,
    photon_fluence_to_collision_air_kerma_rate,
    simulate_surface_source,
)


TEST_CRS = "EPSG:32630"
TEST_GRID_SIZE_M = 20.0
TEST_POLYGON_MIN_M = 5.0
TEST_POLYGON_MAX_M = 15.0
SANITY_GRID_SIZE_M = 400.0

RTOL = 1e-10
ATOL = 1e-12
CONVERSION_RTOL = 1e-15
CONVERSION_ATOL = 0.0

REFERENCE_LOW_KERMA_UGY_H = 0.172
REFERENCE_MEDIUM_KERMA_UGY_H = 1.72
REFERENCE_HIGH_KERMA_UGY_H = 8.6


def _grid(size_m: float = TEST_GRID_SIZE_M) -> GridSpec:
    return GridSpec.from_bounds(
        (0.0, 0.0, size_m, size_m),
        TEST_CRS,
        resolution_m=RASTER_RESOLUTION_M,
    )


def _simulate(activity_density_bq_m2: float):
    grid = _grid()
    source = UniformPolygonSource(
        UniformPolygonConfig(
            source_id="cs137-scenario",
            geometry=box(
                TEST_POLYGON_MIN_M,
                TEST_POLYGON_MIN_M,
                TEST_POLYGON_MAX_M,
                TEST_POLYGON_MAX_M,
            ),
            activity_density_bq_m2=activity_density_bq_m2,
            crs=TEST_CRS,
        )
    )
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(),
        grid,
    )
    return simulate_surface_source(source, kernel, nominal_grid=grid)


def _central_region_mean(array: np.ndarray) -> float:
    center = array.shape[0] // 2
    return float(array[center - 1 : center + 1, center - 1 : center + 1].mean())


def test_fixed_cs137_profile_and_photon_kernel() -> None:
    config = SurfacePhotonResponseConfig()
    grid = _grid()
    kernel = SurfacePhotonResponseKernel.create(config, grid)
    center_row, center_column = kernel.center_index
    expected_center_response = (
        GAMMA_YIELD_PER_DECAY
        * np.exp(-MU_AIR_M_INV * OBSERVATION_HEIGHT_M)
        / (4.0 * np.pi * OBSERVATION_HEIGHT_M**2)
    )

    assert RADIONUCLIDE_NAME == "Cs-137"
    assert HALF_LIFE_YEARS == 30.007
    assert GAMMA_ENERGY_MEV == 0.661657
    assert GAMMA_YIELD_PER_DECAY == 0.851
    assert OBSERVATION_HEIGHT_M == 1.0
    assert RASTER_RESOLUTION_M == 1.0
    assert AIR_DENSITY_KG_M3 == 1.205
    assert MASS_ATTENUATION_COEFF_AIR_M2_KG == 7.752572415e-3
    assert MU_AIR_M_INV == MASS_ATTENUATION_COEFF_AIR_M2_KG * AIR_DENSITY_KG_M3
    assert MASS_ENERGY_ABSORPTION_COEFF_AIR_M2_KG == 2.931111765e-3
    assert MEV_TO_J == 1.602176634e-13
    assert SECONDS_PER_HOUR == 3600.0
    assert GY_TO_UGY == 1.0e6
    assert np.isclose(
        kernel.values[center_row, center_column],
        expected_center_response,
        rtol=RTOL,
        atol=ATOL,
    )
    assert kernel.to_metadata()["kernel_unit"] == "photons/m^2/decay"
    assert kernel.to_metadata()["normalised"] is False
    assert kernel.to_metadata()["kernel_cutoff"] is None


def test_photon_fluence_to_collision_air_kerma_conversion() -> None:
    one_photon_result = photon_fluence_to_collision_air_kerma_rate(1.0)
    million_photon_result = photon_fluence_to_collision_air_kerma_rate(1.0e6)

    assert np.isclose(
        one_photon_result,
        1.1186086791268144e-6,
        rtol=CONVERSION_RTOL,
        atol=CONVERSION_ATOL,
    )
    assert np.isclose(
        million_photon_result,
        1.1186086791268144,
        rtol=CONVERSION_RTOL,
        atol=CONVERSION_ATOL,
    )
    assert one_photon_result == CS137_FLUENCE_TO_KERMA_UGY_H
    print(
        "Cs-137 conversion: 1 photon/m^2/s -> "
        f"{one_photon_result:.16e} uGy/h"
    )
    print(
        "Cs-137 conversion: 1e6 photons/m^2/s -> "
        f"{million_photon_result:.16e} uGy/h"
    )


def test_low_medium_high_activity_scenarios_scale_linearly() -> None:
    scenario_results = {
        "LOW": _simulate(LOW_ACTIVITY_DENSITY_BQ_M2),
        "MEDIUM": _simulate(MEDIUM_ACTIVITY_DENSITY_BQ_M2),
        "HIGH": _simulate(HIGH_ACTIVITY_DENSITY_BQ_M2),
    }
    fluence_maxima = {
        name: float(result.photon_fluence_rate_patch.photon_fluence_rate.max())
        for name, result in scenario_results.items()
    }
    kerma_maxima = {
        name: float(
            result.collision_air_kerma_rate_patch
            .collision_air_kerma_rate_uGy_h.max()
        )
        for name, result in scenario_results.items()
    }

    assert np.isclose(
        fluence_maxima["MEDIUM"] / fluence_maxima["LOW"],
        10.0,
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.isclose(
        fluence_maxima["HIGH"] / fluence_maxima["LOW"],
        50.0,
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.isclose(
        kerma_maxima["MEDIUM"] / kerma_maxima["LOW"],
        10.0,
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.isclose(
        kerma_maxima["HIGH"] / kerma_maxima["LOW"],
        50.0,
        rtol=RTOL,
        atol=ATOL,
    )

    for name, activity_density in (
        ("LOW", LOW_ACTIVITY_DENSITY_BQ_M2),
        ("MEDIUM", MEDIUM_ACTIVITY_DENSITY_BQ_M2),
        ("HIGH", HIGH_ACTIVITY_DENSITY_BQ_M2),
    ):
        kerma_map = (
            scenario_results[name]
            .collision_air_kerma_rate_patch
            .collision_air_kerma_rate_uGy_h
        )
        print(
            f"{name}: activity={activity_density:.6e} Bq/m^2, "
            f"max fluence={fluence_maxima[name]:.12e} photons/m^2/s, "
            f"max kerma={kerma_maxima[name]:.12e} uGy/h, "
            f"central kerma={_central_region_mean(kerma_map):.12e} uGy/h"
        )


def test_kerma_map_is_strictly_linear_in_photon_fluence() -> None:
    result = _simulate(MEDIUM_ACTIVITY_DENSITY_BQ_M2)
    fluence_map = result.photon_fluence_rate_patch.photon_fluence_rate
    kerma_map = (
        result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h
    )
    expected_kerma = fluence_map * CS137_FLUENCE_TO_KERMA_UGY_H

    assert np.allclose(
        kerma_map,
        expected_kerma,
        rtol=CONVERSION_RTOL,
        atol=CONVERSION_ATOL,
    )
    assert result.collision_air_kerma_metadata["output_unit"] == (
        COLLISION_AIR_KERMA_RATE_UNIT
    )
    assert result.collision_air_kerma_metadata[
        "gamma_yield_applied_in_kerma_conversion"
    ] is False
    assert result.collision_air_kerma_metadata["dose_equivalent"] is False
    assert result.collision_air_kerma_metadata["effective_dose"] is False


def test_raw_fft_negative_values_are_only_roundoff() -> None:
    result = _simulate(MEDIUM_ACTIVITY_DENSITY_BQ_M2)
    minimum_raw = result.minimum_raw_fft_value_before_clipping
    maximum_physical = float(
        result.photon_fluence_rate_patch.photon_fluence_rate.max()
    )
    floating_point_bound = (
        np.finfo(np.float64).eps
        * max(1.0, maximum_physical)
        * (result.cell_activity_bq.size + result.kernel.values.size)
    )

    assert minimum_raw >= -floating_point_bound
    print(f"Minimum raw FFT value before clipping: {minimum_raw:.16e}")
    if minimum_raw < 0.0:
        print(f"Negative raw FFT magnitude: {abs(minimum_raw):.16e}")
    else:
        print("Negative raw FFT value observed: NO")
    print(f"Derived floating-point round-off bound: {floating_point_bound:.16e}")


def test_tiny_negative_fft_roundoff_is_recorded_and_clipped(monkeypatch) -> None:
    negative_roundoff = -np.finfo(np.float64).eps

    def fake_fftconvolve(cell_activity, kernel_values, mode):
        assert mode == "same"
        assert kernel_values.ndim == 2
        raw = np.ones_like(cell_activity, dtype=np.float64)
        raw[0, 0] = negative_roundoff
        return raw

    monkeypatch.setattr(surface_simulator, "fftconvolve", fake_fftconvolve)
    result = _simulate(LOW_ACTIVITY_DENSITY_BQ_M2)
    fluence = result.photon_fluence_rate_patch.photon_fluence_rate
    kerma = result.collision_air_kerma_rate_patch.collision_air_kerma_rate_uGy_h

    assert result.minimum_raw_fft_value_before_clipping == negative_roundoff
    assert fluence[0, 0] == 0.0
    assert kerma[0, 0] == 0.0


def test_large_uniform_surface_sanity_diagnostic() -> None:
    grid = _grid(SANITY_GRID_SIZE_M)
    kernel = SurfacePhotonResponseKernel.create(
        SurfacePhotonResponseConfig(),
        grid,
    )
    reference_values = {
        "LOW": REFERENCE_LOW_KERMA_UGY_H,
        "MEDIUM": REFERENCE_MEDIUM_KERMA_UGY_H,
        "HIGH": REFERENCE_HIGH_KERMA_UGY_H,
    }
    scenario_values = (
        ("LOW", LOW_ACTIVITY_DENSITY_BQ_M2),
        ("MEDIUM", MEDIUM_ACTIVITY_DENSITY_BQ_M2),
        ("HIGH", HIGH_ACTIVITY_DENSITY_BQ_M2),
    )
    model_values: dict[str, float] = {}

    for name, activity_density in scenario_values:
        source = UniformPolygonSource(
            UniformPolygonConfig(
                source_id=f"cs137-large-{name.lower()}",
                geometry=box(0.0, 0.0, SANITY_GRID_SIZE_M, SANITY_GRID_SIZE_M),
                activity_density_bq_m2=activity_density,
                crs=TEST_CRS,
            )
        )
        result = simulate_surface_source(source, kernel, nominal_grid=grid)
        center = grid.height // 2
        model_value = float(
            result.collision_air_kerma_rate_patch
            .collision_air_kerma_rate_uGy_h[center, center]
        )
        model_values[name] = model_value
        reference_value = reference_values[name]
        print(
            f"Sanity {name}: model={model_value:.12e} uGy/h, "
            f"external reference={reference_value:.12e} uGy/h, "
            f"model/reference={model_value / reference_value:.12e}"
        )

    assert all(value > 0.0 for value in model_values.values())
    assert np.isclose(
        model_values["MEDIUM"] / model_values["LOW"],
        10.0,
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.isclose(
        model_values["HIGH"] / model_values["LOW"],
        50.0,
        rtol=RTOL,
        atol=ATOL,
    )
