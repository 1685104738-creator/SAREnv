import math

import numpy as np
import pytest

from sarenv.analytics.radiation_priority import (
    GreedyCandidate,
    RadiationPriorityGreedyPolicy,
    SAR_PROBABILITY_GATE,
    estimate_candidate_radiation,
    greedy_candidate_from_grid,
)
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.measurement import RadiationEstimate, RadiationMeasurement


CRS = "EPSG:32630"
QUANTITY = "synthetic_radiation_rate"
UNIT = "benchmark_unit"


class StubEstimator:
    def __init__(self, values_by_position):
        self.values_by_position = values_by_position
        self.queries = []

    def estimate_at(self, x_m, y_m):
        self.queries.append((x_m, y_m))
        value = self.values_by_position.get((x_m, y_m))
        return RadiationEstimate(
            x_m=x_m,
            y_m=y_m,
            value=value,
            weight_sum=0.0 if value is None else 1.0,
            quantity=QUANTITY,
            unit=UNIT,
        )


def _measurement(x_m: float, y_m: float, value: float = 8.0):
    return RadiationMeasurement(
        x_m=x_m,
        y_m=y_m,
        altitude_m=50.0,
        simulated_time_s=0.0,
        value=value,
        quantity=QUANTITY,
        unit=UNIT,
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


def test_approved_roi_supports_immediate_30_m_greedy_neighbours():
    grid = GridSpec.from_bounds((0.0, 0.0, 120.0, 120.0), CRS)
    estimator = IncrementalRadiationGrid(
        grid,
        quantity=QUANTITY,
        unit=UNIT,
    )
    estimator.update(_measurement(45.0, 45.0))

    orthogonal = greedy_candidate_from_grid(
        1,
        2,
        0.2,
        bounds=(0.0, 0.0, 90.0, 90.0),
        map_shape=(3, 3),
    )
    diagonal = greedy_candidate_from_grid(
        2,
        2,
        0.1,
        bounds=(0.0, 0.0, 90.0, 90.0),
        map_shape=(3, 3),
    )

    assert math.dist((45.0, 45.0), (orthogonal.x_m, orthogonal.y_m)) == 30.0
    assert math.dist((45.0, 45.0), (diagonal.x_m, diagonal.y_m)) == pytest.approx(
        math.sqrt(2.0) * 30.0
    )
    assert estimate_candidate_radiation(orthogonal, estimator).observed
    assert estimate_candidate_radiation(diagonal, estimator).observed
    assert estimate_candidate_radiation(orthogonal, estimator).value == 8.0
    assert estimate_candidate_radiation(diagonal, estimator).value == 8.0


def test_candidate_outside_local_support_remains_explicitly_unknown():
    grid = GridSpec.from_bounds((0.0, 0.0, 150.0, 150.0), CRS)
    estimator = IncrementalRadiationGrid(
        grid,
        quantity=QUANTITY,
        unit=UNIT,
    )
    estimator.update(_measurement(45.0, 45.0))
    unsupported = greedy_candidate_from_grid(
        1,
        3,
        0.1,
        bounds=(0.0, 0.0, 150.0, 90.0),
        map_shape=(3, 5),
    )

    estimate = estimate_candidate_radiation(unsupported, estimator)

    assert unsupported.x_m == 105.0
    assert estimate.value is None
    assert not estimate.observed


def test_multiple_spatial_measurements_create_local_directional_ordering():
    bounds = (0.0, 0.0, 150.0, 150.0)
    grid = GridSpec.from_bounds(bounds, CRS)
    estimator = IncrementalRadiationGrid(
        grid,
        quantity=QUANTITY,
        unit=UNIT,
    )
    estimator.update(_measurement(15.0, 75.0, value=1.0))
    estimator.update(_measurement(45.0, 75.0, value=2.0))
    estimator.update(_measurement(75.0, 75.0, value=3.0))

    candidates = [
        greedy_candidate_from_grid(
            row,
            col,
            0.1,
            bounds=bounds,
            map_shape=(5, 5),
        )
        for row in (1, 2, 3)
        for col in (1, 2, 3)
        if (row, col) != (2, 2)
    ]
    estimates = {
        (candidate.row, candidate.col): estimate_candidate_radiation(
            candidate, estimator
        )
        for candidate in candidates
    }

    assert all(estimate.observed for estimate in estimates.values())
    for row in (1, 2, 3):
        assert estimates[(row, 3)].value > estimates[(row, 1)].value


def _candidate(col: int, sar_score: float) -> GreedyCandidate:
    return GreedyCandidate(
        row=0,
        col=col,
        x_m=float(col),
        y_m=0.0,
        sar_score=sar_score,
    )


def test_policy_falls_back_without_confirmed_radiation_evidence():
    policy = RadiationPriorityGreedyPolicy()
    estimator = StubEstimator({})

    decision = policy.select(
        [_candidate(0, 0.5)],
        estimator,
        radiation_priority_enabled=False,
    )

    assert SAR_PROBABILITY_GATE == 0.001
    assert decision.fallback_to_original
    assert decision.reason == "radiation_not_confirmed"
    assert estimator.queries == []


def test_policy_applies_sar_gate_then_radiation_then_sar_tie_break():
    below_gate = _candidate(0, 0.0009)
    lower_radiation = _candidate(1, 0.4)
    higher_radiation_lower_sar = _candidate(2, 0.1)
    estimator = StubEstimator(
        {
            (below_gate.x_m, below_gate.y_m): 100.0,
            (lower_radiation.x_m, lower_radiation.y_m): 2.0,
            (higher_radiation_lower_sar.x_m, higher_radiation_lower_sar.y_m): 3.0,
        }
    )

    decision = RadiationPriorityGreedyPolicy().select(
        [below_gate, lower_radiation, higher_radiation_lower_sar],
        estimator,
        radiation_priority_enabled=True,
    )

    assert not decision.fallback_to_original
    assert decision.selected_candidate is higher_radiation_lower_sar
    assert decision.radiation_estimate.value == 3.0
    assert (below_gate.x_m, below_gate.y_m) not in estimator.queries

    equal_radiation = StubEstimator(
        {
            (lower_radiation.x_m, lower_radiation.y_m): 3.0,
            (higher_radiation_lower_sar.x_m, higher_radiation_lower_sar.y_m): 3.0,
        }
    )
    tie_decision = RadiationPriorityGreedyPolicy().select(
        [lower_radiation, higher_radiation_lower_sar],
        equal_radiation,
        radiation_priority_enabled=True,
    )
    assert tie_decision.selected_candidate is lower_radiation


def test_policy_falls_back_when_all_or_part_of_eligible_estimates_are_unknown():
    first = _candidate(1, 0.4)
    second = _candidate(2, 0.3)
    policy = RadiationPriorityGreedyPolicy()

    all_unknown = policy.select(
        [first, second],
        StubEstimator({}),
        radiation_priority_enabled=True,
    )
    partial = policy.select(
        [first, second],
        StubEstimator({(first.x_m, first.y_m): 2.0}),
        radiation_priority_enabled=True,
    )

    assert all_unknown.fallback_to_original
    assert all_unknown.reason == "no_usable_radiation_estimate"
    assert partial.fallback_to_original
    assert partial.reason == "partial_radiation_estimates"


def test_policy_never_creates_a_candidate_outside_its_input():
    candidates = [_candidate(1, 0.4), _candidate(2, 0.3)]
    estimator = StubEstimator({(1.0, 0.0): 2.0, (2.0, 0.0): 3.0})

    decision = RadiationPriorityGreedyPolicy().select(
        candidates,
        estimator,
        radiation_priority_enabled=True,
    )

    assert any(decision.selected_candidate is candidate for candidate in candidates)
