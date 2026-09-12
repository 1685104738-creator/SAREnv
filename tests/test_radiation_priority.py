import math

import numpy as np
import pytest

from sarenv.analytics.radiation_priority import (
    RadiationPriorityGreedyPolicy,
    derive_sar_reference_score,
    estimate_candidate_radiation,
    greedy_candidate_from_grid,
)
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.measurement import (
    RadiationMeasurement,
    RadiationRegionEstimate,
)


CRS = "EPSG:32630"
QUANTITY = "synthetic_excess_gamma_dose_rate"
UNIT = "uSv/h"


class StubEstimator:
    def __init__(self, values_by_position):
        self.values_by_position = values_by_position
        self.queries = []

    def max_estimate_in_bounds(self, bounds):
        x_m = (bounds[0] + bounds[2]) / 2.0
        y_m = (bounds[1] + bounds[3]) / 2.0
        self.queries.append((x_m, y_m))
        value, supported = self.values_by_position.get((x_m, y_m), (0.0, False))
        return RadiationRegionEstimate(
            bounds=bounds,
            value=value,
            max_weight_sum=1.0 if supported else 0.0,
            supported_cell_count=1 if supported else 0,
            total_cell_count=1,
            quantity=QUANTITY,
            unit=UNIT,
        )


def _measurement(x_m: float, y_m: float, value: float = 8.0):
    return RadiationMeasurement(
        x_m=x_m,
        y_m=y_m,
        platform_altitude_m=50.0,
        value_reference_height_m=50.0,
        simulated_time_s=0.0,
        value=value,
        quantity=QUANTITY,
        unit=UNIT,
    )


def _candidate(col: int, sar_score: float):
    return greedy_candidate_from_grid(
        0,
        col,
        sar_score,
        bounds=(0.0, 0.0, 3.0, 1.0),
        map_shape=(1, 3),
    )


def test_candidate_centre_matches_original_greedy_coordinate_mapping():
    candidate = greedy_candidate_from_grid(
        1,
        2,
        0.25,
        bounds=(0.0, 0.0, 90.0, 90.0),
        map_shape=(3, 3),
    )

    assert (candidate.x_m, candidate.y_m) == (75.0, 45.0)
    assert candidate.sar_score == pytest.approx(0.25)


def test_candidate_accepts_numpy_integer_indices_used_by_original_greedy():
    candidate = greedy_candidate_from_grid(
        np.int64(1),
        np.int64(2),
        0.25,
        bounds=(0.0, 0.0, 90.0, 90.0),
        map_shape=(3, 3),
    )

    assert type(candidate.row) is int
    assert type(candidate.col) is int


def test_candidate_radiation_is_max_over_full_sar_cell():
    bounds = (0.0, 0.0, 60.0, 60.0)
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(bounds, CRS),
        quantity=QUANTITY,
        unit=UNIT,
        update_radius_m=0.49,
        kernel_sigma_m=0.2,
    )
    estimator.update(_measurement(25.5, 25.5, 8.0))
    candidate = greedy_candidate_from_grid(
        1,
        1,
        0.2,
        bounds=bounds,
        map_shape=(3, 3),
    )

    estimate = estimate_candidate_radiation(
        candidate,
        estimator,
        bounds=bounds,
        map_shape=(3, 3),
    )

    assert estimate.value == 8.0
    assert estimate.total_cell_count == 400
    assert estimate.supported_cell_count == 1


def test_unsupported_candidate_is_numeric_zero_not_unknown():
    bounds = (0.0, 0.0, 60.0, 60.0)
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(bounds, CRS),
        quantity=QUANTITY,
        unit=UNIT,
    )
    candidate = greedy_candidate_from_grid(
        1,
        1,
        0.2,
        bounds=bounds,
        map_shape=(3, 3),
    )

    estimate = estimate_candidate_radiation(
        candidate,
        estimator,
        bounds=bounds,
        map_shape=(3, 3),
    )

    assert estimate.value == 0.0
    assert not estimate.observed


def test_reference_score_uses_native_visible_probability_on_half_radius_ring():
    probability_map = np.full((6, 6), 1.0 / 36.0)

    reference = derive_sar_reference_score(
        probability_map=probability_map,
        bounds=(0.0, 0.0, 120.0, 120.0),
        search_center=(60.0, 60.0),
        max_radius=60.0,
        detection_radius_m=20.71,
        reference_radius_fraction=0.5,
    )

    assert reference.radius_m == 30.0
    assert reference.ring_cell_count > 0
    assert reference.score == pytest.approx(5.0 / 36.0)
    assert reference.method == "minimum_native_visible_probability_on_reference_ring"


def test_policy_uses_sar_reference_then_radiation_equivalent_priority():
    below_reference = _candidate(0, 0.09)
    lower_radiation = _candidate(1, 0.4)
    higher_radiation = _candidate(2, 0.2)
    estimator = StubEstimator(
        {
            (0.5, 0.5): (100.0, False),
            (1.5, 0.5): (2.0, True),
            (2.5, 0.5): (3.0, True),
        }
    )
    policy = RadiationPriorityGreedyPolicy(
        sar_reference_score=0.1,
        hazard_reference_excess=2.0,
    )

    decision = policy.select(
        [below_reference, lower_radiation, higher_radiation],
        estimator,
        radiation_priority_enabled=True,
        bounds=(0.0, 0.0, 3.0, 1.0),
        map_shape=(1, 3),
    )

    assert decision.selected_candidate is higher_radiation
    selected = next(item for item in decision.assessments if item.chosen)
    assert selected.radiation_estimate.value == 3.0
    assert selected.radiation_equivalent_priority == pytest.approx(0.15)
    assert not decision.assessments[0].sar_relevant


def test_policy_numeric_zero_estimates_do_not_force_unknown_fallback():
    first = _candidate(1, 0.4)
    second = _candidate(2, 0.3)
    policy = RadiationPriorityGreedyPolicy(
        sar_reference_score=0.1,
        hazard_reference_excess=2.0,
    )

    decision = policy.select(
        [first, second],
        StubEstimator({}),
        radiation_priority_enabled=True,
        bounds=(0.0, 0.0, 3.0, 1.0),
        map_shape=(1, 3),
    )

    assert not decision.fallback_to_original
    assert decision.selected_candidate is first
    assert decision.radiation_estimate.value == 0.0


def test_policy_never_creates_a_candidate_outside_its_input():
    candidates = [_candidate(1, 0.4), _candidate(2, 0.3)]
    estimator = StubEstimator({(1.5, 0.5): (2.0, True), (2.5, 0.5): (3.0, True)})
    policy = RadiationPriorityGreedyPolicy(
        sar_reference_score=0.1,
        hazard_reference_excess=2.0,
    )

    decision = policy.select(
        candidates,
        estimator,
        radiation_priority_enabled=True,
        bounds=(0.0, 0.0, 3.0, 1.0),
        map_shape=(1, 3),
    )

    assert any(decision.selected_candidate is candidate for candidate in candidates)
    assert math.isfinite(decision.radiation_estimate.value)
