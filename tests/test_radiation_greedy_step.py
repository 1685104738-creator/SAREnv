import inspect

import numpy as np

from sarenv.analytics.paths import greedy_visible_cells
from sarenv.analytics.radiation_priority import (
    RadiationGreedyStepPlanner,
    SarReferenceScore,
)
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.measurement import RadiationMeasurement


CRS = "EPSG:32630"
BOUNDS = (0.0, 0.0, 150.0, 150.0)
QUANTITY = "test_radiation_rate"
UNIT = "test_unit"


def _measurement(x_m, value):
    return RadiationMeasurement(
        x_m=x_m,
        y_m=75.0,
        altitude_m=50.0,
        simulated_time_s=0.0,
        value=value,
        quantity=QUANTITY,
        unit=UNIT,
    )


def _planner(probability_map, estimator):
    return RadiationGreedyStepPlanner(
        probability_map=probability_map,
        bounds=BOUNDS,
        search_center=(75.0, 75.0),
        max_radius=200.0,
        detection_radius_m=0.0,
        estimator=estimator,
    )


def _directional_estimator():
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(BOUNDS, CRS),
        quantity=QUANTITY,
        unit=UNIT,
    )
    estimator.update(_measurement(15.0, 1.0))
    estimator.update(_measurement(45.0, 2.0))
    estimator.update(_measurement(75.0, 3.0))
    return estimator


def test_radiation_greedy_returns_exactly_one_existing_neighbour():
    probability_map = np.full((5, 5), 0.01)
    probability_map[2, 3] = 0.02
    estimator = _directional_estimator()
    observed = greedy_visible_cells(
        2,
        2,
        map_shape=probability_map.shape,
        bounds=BOUNDS,
        detection_radius_m=0.0,
    )

    result = _planner(probability_map, estimator).plan_next(
        (2, 2),
        observed,
        radiation_priority_enabled=True,
    )

    assert len(result.candidates) == 8
    assert result.selected_candidate is not None
    assert (result.selected_candidate.row, result.selected_candidate.col) == (2, 3)
    assert any(
        result.selected_candidate is candidate for candidate in result.candidates
    )
    assert result.planning_time_s >= 0.0
    assert not result.normal_replan_required
    assert observed == frozenset({(2, 2)})


def test_radiation_step_uses_numeric_zero_without_unknown_fallback():
    probability_map = np.full((5, 5), 0.01)
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(BOUNDS, CRS),
        quantity=QUANTITY,
        unit=UNIT,
    )

    result = _planner(probability_map, estimator).plan_next(
        (2, 2),
        set(),
        radiation_priority_enabled=True,
    )

    assert result.selected_candidate is not None
    assert not result.normal_replan_required
    assert result.decision.reason == "radiation_priority"
    assert result.decision.radiation_estimate.value == 0.0


def test_no_sar_relevant_candidate_takes_one_original_greedy_fallback_step():
    probability_map = np.full((5, 5), 0.0009)
    probability_map[2, 1] = 0.002
    estimator = _directional_estimator()

    planner = RadiationGreedyStepPlanner(
        probability_map=probability_map,
        bounds=BOUNDS,
        search_center=(75.0, 75.0),
        max_radius=200.0,
        detection_radius_m=0.0,
        estimator=estimator,
        hazard_reference_excess=1.0,
        sar_reference=SarReferenceScore(
            score=0.01,
            radius_fraction=0.5,
            radius_m=100.0,
            ring_cell_count=1,
        ),
        rng=np.random.default_rng(7),
    )

    result = planner.plan_next(
        (2, 2),
        {(2, 2)},
        radiation_priority_enabled=True,
    )

    assert result.selected_candidate is not None
    assert (result.selected_candidate.row, result.selected_candidate.col) == (2, 1)
    assert result.selected_candidate.sar_score == 0.002
    assert result.decision.fallback_to_original
    assert result.decision.reason.endswith("original_greedy_step")


def test_step_planner_has_no_radiation_truth_parameter():
    constructor_parameters = inspect.signature(
        RadiationGreedyStepPlanner.__init__
    ).parameters
    planning_parameters = inspect.signature(
        RadiationGreedyStepPlanner.plan_next
    ).parameters

    assert "truth" not in constructor_parameters
    assert "radiation_truth" not in constructor_parameters
    assert "truth" not in planning_parameters
    assert "radiation_truth" not in planning_parameters
