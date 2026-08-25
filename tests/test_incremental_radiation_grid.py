"""Tests for the fixed-ROI incremental radiation estimator."""

import math

import numpy as np
import pytest

from sarenv.radiation import GridSpec
from sarenv.radiation.online import (
    IncrementalRadiationGrid,
    RADIATION_KERNEL_SIGMA_M,
    RADIATION_UPDATE_RADIUS_M,
    RadiationMeasurement,
)


CRS = "EPSG:32630"
QUANTITY = "synthetic_radiation_scale"
UNIT = "benchmark_unit"


def _measurement(x_m: float, y_m: float, value: float) -> RadiationMeasurement:
    return RadiationMeasurement(
        x_m=x_m,
        y_m=y_m,
        altitude_m=50.0,
        simulated_time_s=0.0,
        value=value,
        quantity=QUANTITY,
        unit=UNIT,
    )


def _estimator(size_m: int = 200) -> IncrementalRadiationGrid:
    grid = GridSpec.from_bounds((0.0, 0.0, size_m, size_m), CRS)
    return IncrementalRadiationGrid(grid, quantity=QUANTITY, unit=UNIT)


def test_baseline_radius_and_sigma_are_the_approved_values() -> None:
    estimator = _estimator()

    assert estimator.update_radius_m == RADIATION_UPDATE_RADIUS_M == 45.0
    assert estimator.kernel_sigma_m == RADIATION_KERNEL_SIGMA_M == 15.0


def test_first_measurement_updates_only_the_circular_local_roi() -> None:
    estimator = _estimator()
    updated_cells = estimator.update(_measurement(100.5, 100.5, 7.0))

    expected_cells = sum(
        dx * dx + dy * dy <= 45 * 45
        for dx in range(-45, 46)
        for dy in range(-45, 46)
    )
    assert updated_cells == expected_cells
    assert estimator.estimate_at(100.5, 100.5).value == 7.0
    assert estimator.estimate_at(145.5, 100.5).observed
    assert not estimator.estimate_at(146.5, 100.5).observed
    assert estimator.mean[100, 146] == 0.0
    assert estimator.weight_sum[100, 146] == 0.0


def test_repeated_measurements_use_incremental_weighted_mean() -> None:
    estimator = _estimator()
    estimator.update(_measurement(100.5, 100.5, 4.0))
    estimator.update(_measurement(100.5, 100.5, 10.0))

    estimate = estimator.estimate_at(100.5, 100.5)
    assert estimate.value == pytest.approx(7.0)
    assert estimate.weight_sum == pytest.approx(2.0)


def test_multiple_measurements_use_gaussian_distance_weights() -> None:
    estimator = _estimator()
    estimator.update(_measurement(100.5, 100.5, 10.0))
    estimator.update(_measurement(115.5, 100.5, 20.0))

    second_weight = math.exp(-0.5 * (15.0 / 15.0) ** 2)
    expected = (10.0 + second_weight * 20.0) / (1.0 + second_weight)
    estimate = estimator.estimate_at(100.5, 100.5)
    assert estimate.value == pytest.approx(expected)
    assert estimate.weight_sum == pytest.approx(1.0 + second_weight)


def test_unsupported_cells_remain_numeric_zero_with_zero_weight() -> None:
    estimator = _estimator()
    estimator.update(_measurement(25.5, 25.5, 3.0))

    estimate = estimator.estimate_at(150.5, 150.5)
    assert estimate.value == 0.0
    assert estimate.weight_sum == 0.0
    assert not estimate.observed


def test_region_query_uses_max_and_dynamic_overlapping_cell_bounds() -> None:
    grid = GridSpec(
        bounds=(0.25, 0.25, 10.25, 10.25),
        resolution_m=1.0,
        crs=CRS,
        width=10,
        height=10,
    )
    estimator = IncrementalRadiationGrid(
        grid,
        quantity=QUANTITY,
        unit=UNIT,
        update_radius_m=0.49,
        kernel_sigma_m=0.2,
    )
    estimator.update(_measurement(4.75, 4.75, 3.0))
    estimator.update(_measurement(5.75, 5.75, 9.0))

    region = estimator.max_estimate_in_bounds((4.25, 4.25, 6.25, 6.25))

    assert region.value == 9.0
    assert region.total_cell_count == 4
    assert region.supported_cell_count == 2
    assert region.support_fraction == 0.5


def test_region_query_outside_grid_returns_numeric_zero() -> None:
    estimator = _estimator()

    region = estimator.max_estimate_in_bounds((300.0, 300.0, 310.0, 310.0))

    assert region.value == 0.0
    assert region.total_cell_count == 0
    assert region.supported_cell_count == 0


def test_estimator_stores_fixed_arrays_not_measurement_history() -> None:
    estimator = _estimator()
    for index in range(20):
        estimator.update(_measurement(100.5, 100.5, float(index)))

    array_state = [
        value for value in vars(estimator).values() if isinstance(value, np.ndarray)
    ]
    sequence_state = [
        value
        for value in vars(estimator).values()
        if isinstance(value, (list, tuple, set, dict))
    ]
    assert len(array_state) == 2
    assert all(array.shape == estimator.grid.shape for array in array_state)
    assert sequence_state == []


def test_estimator_rejects_quantity_or_unit_mismatch() -> None:
    estimator = _estimator()
    measurement = RadiationMeasurement(
        x_m=100.5,
        y_m=100.5,
        altitude_m=50.0,
        simulated_time_s=0.0,
        value=1.0,
        quantity=QUANTITY,
        unit="uSv/h",
    )

    with pytest.raises(ValueError, match="quantity/unit"):
        estimator.update(measurement)
