import numpy as np
import pytest
from shapely.geometry import LineString

from sarenv.analytics.radiation_priority import RadiationGreedyStepPlanner
from sarenv.analytics.route_execution import (
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
    RadiationGreedyExecutionController,
    execute_normal_route_node_with_radiation,
    resume_normal_route_after_radiation,
)
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.sensor import NoiseFreeRadiationSensor
from sarenv.radiation.online.trigger import (
    AboveBackgroundCriterion,
    ConsecutiveRadiationTrigger,
)


CRS = "EPSG:32630"
BOUNDS = (0.0, 0.0, 150.0, 150.0)
QUANTITY = "test_radiation_rate"
UNIT = "test_unit"


def _pipeline(probability_map):
    state = ExecutedSARState(
        map_shape=probability_map.shape,
        bounds=BOUNDS,
        detection_radius_m=0.0,
    )
    normal_route = PrecomputedRouteController(
        LineString(
            [
                (15.0, 75.0),
                (45.0, 75.0),
                (75.0, 75.0),
                (105.0, 75.0),
                (135.0, 75.0),
            ]
        ),
        state,
    )
    queried_positions = []

    def test_only_truth(x_m, y_m, altitude_m):
        queried_positions.append((x_m, y_m, altitude_m))
        return x_m / 30.0

    sensor = NoiseFreeRadiationSensor(
        test_only_truth,
        quantity=QUANTITY,
        unit=UNIT,
    )
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(BOUNDS, CRS),
        quantity=QUANTITY,
        unit=UNIT,
    )
    trigger = ConsecutiveRadiationTrigger(
        AboveBackgroundCriterion(0.0, quantity=QUANTITY, unit=UNIT)
    )
    observer = ExecutedNodeRadiationObserver(sensor, estimator, trigger)
    planner = RadiationGreedyStepPlanner(
        probability_map=probability_map,
        bounds=BOUNDS,
        search_center=(75.0, 75.0),
        max_radius=200.0,
        detection_radius_m=0.0,
        estimator=estimator,
    )
    return state, normal_route, queried_positions, observer, planner


def test_core_flow_interrupts_normal_route_then_executes_one_radiation_step():
    probability_map = np.full((5, 5), 0.01)
    probability_map[2, 3] = 0.02
    state, normal_route, queries, observer, planner = _pipeline(probability_map)

    for test_only_time_s in (0.0, 1.0, 2.0):
        normal_result = execute_normal_route_node_with_radiation(
            normal_route,
            observer,
            altitude_m=50.0,
            simulated_time_s=test_only_time_s,
        )

    assert normal_result.normal_route_interrupted
    assert normal_route.invalidated
    assert normal_route.discarded_node_count == 2
    assert state.current_position_m == (75.0, 75.0)
    assert (2, 3) not in state.observed_cells

    radiation_controller = RadiationGreedyExecutionController(
        state,
        planner,
        observer,
        initial_budget_m=300.0,  # TEST ONLY: not a production mission default.
    )
    result = radiation_controller.execute_next(
        altitude_m=50.0,
        simulated_time_s=3.0,
    )

    assert result.normal_route_replan is None
    assert (result.selected_candidate.row, result.selected_candidate.col) == (2, 3)
    assert result.route_execution.distance_from_previous_m == pytest.approx(30.0)
    assert state.current_position_m == (105.0, 75.0)
    assert state.total_executed_distance_m == pytest.approx(90.0)
    assert (2, 3) in state.observed_cells
    assert state.executed_positions_m == (
        (15.0, 75.0),
        (45.0, 75.0),
        (75.0, 75.0),
        (105.0, 75.0),
    )
    assert queries == [
        (15.0, 75.0, 50.0),
        (45.0, 75.0, 50.0),
        (75.0, 75.0, 50.0),
        (105.0, 75.0, 50.0),
    ]


def test_no_gate_eligible_candidate_requests_normal_replan_without_moving():
    probability_map = np.full((5, 5), 0.0009)
    state, normal_route, _, observer, planner = _pipeline(probability_map)
    for test_only_time_s in (0.0, 1.0, 2.0):
        execute_normal_route_node_with_radiation(
            normal_route,
            observer,
            altitude_m=50.0,
            simulated_time_s=test_only_time_s,
        )
    position_before = state.current_position_m
    observed_before = state.observed_cells

    result = RadiationGreedyExecutionController(
        state,
        planner,
        observer,
        initial_budget_m=300.0,  # TEST ONLY: not a production mission default.
    ).execute_next(
        altitude_m=50.0,
        simulated_time_s=3.0,
    )

    assert result.selected_candidate is None
    assert result.route_execution is None
    assert result.radiation_observation is None
    assert result.normal_route_replan.current_position_m == position_before
    assert result.normal_route_replan.observed_cells == observed_before
    assert state.current_position_m == position_before
    assert state.observed_cells == observed_before


def test_normal_radiation_normal_cycle_shares_budget_and_rearms_trigger():
    probability_map = np.full((5, 5), 0.0009)
    probability_map[2, 3] = 0.02
    state, old_normal_route, _, observer, planner = _pipeline(probability_map)

    for test_only_time_s in (0.0, 1.0, 2.0):
        execute_normal_route_node_with_radiation(
            old_normal_route,
            observer,
            altitude_m=50.0,
            simulated_time_s=test_only_time_s,
        )
    assert old_normal_route.invalidated
    assert old_normal_route.discarded_node_count == 2
    assert state.total_executed_distance_m == pytest.approx(60.0)

    radiation_controller = RadiationGreedyExecutionController(
        state,
        planner,
        observer,
        initial_budget_m=300.0,  # TEST ONLY: not a production mission default.
    )
    radiation_move = radiation_controller.execute_next(
        altitude_m=50.0,
        simulated_time_s=3.0,
    )
    assert radiation_move.selected_candidate is not None
    assert state.total_executed_distance_m == pytest.approx(90.0)

    radiation_exit = radiation_controller.execute_next(
        altitude_m=50.0,
        simulated_time_s=4.0,
    )
    assert radiation_exit.normal_route_replan is not None
    assert state.total_executed_distance_m == pytest.approx(90.0)

    test_only_initial_budget_m = 300.0
    continuation = resume_normal_route_after_radiation(
        radiation_exit.normal_route_replan,
        state,
        observer,
        initial_budget_m=test_only_initial_budget_m,
        probability_map=probability_map,
        bounds=BOUNDS,
        search_center=(75.0, 75.0),
        max_radius=200.0,
        fov_deg=0.0,
        altitude=50.0,
    )

    assert not continuation.mission_complete
    assert continuation.remaining_budget_m == pytest.approx(210.0)
    assert continuation.route.length <= continuation.remaining_budget_m
    assert continuation.route.coords[0] == state.current_position_m
    assert continuation.controller is not old_normal_route
    assert continuation.controller.next_node_index == 1
    assert old_normal_route.invalidated
    assert not observer.anomaly_confirmed
    assert observer.consecutive_anomaly_count == 0
    assert state.total_executed_distance_m == pytest.approx(90.0)

    first_new_anomaly = execute_normal_route_node_with_radiation(
        continuation.controller,
        observer,
        altitude_m=50.0,
        simulated_time_s=5.0,
    )
    second_new_anomaly = execute_normal_route_node_with_radiation(
        continuation.controller,
        observer,
        altitude_m=50.0,
        simulated_time_s=6.0,
    )

    assert not first_new_anomaly.normal_route_interrupted
    assert not second_new_anomaly.normal_route_interrupted
    assert not observer.anomaly_confirmed
    assert observer.consecutive_anomaly_count == 2

    third_new_anomaly = execute_normal_route_node_with_radiation(
        continuation.controller,
        observer,
        altitude_m=50.0,
        simulated_time_s=7.0,
    )

    assert third_new_anomaly.normal_route_interrupted
    assert observer.anomaly_confirmed


def test_exhausted_global_budget_ends_radiation_mode_without_moving():
    probability_map = np.full((5, 5), 0.01)
    state, normal_route, _, observer, planner = _pipeline(probability_map)
    for test_only_time_s in (0.0, 1.0, 2.0):
        execute_normal_route_node_with_radiation(
            normal_route,
            observer,
            altitude_m=50.0,
            simulated_time_s=test_only_time_s,
        )
    assert state.total_executed_distance_m == pytest.approx(60.0)

    position_before = state.current_position_m
    observed_before = state.observed_cells
    result = RadiationGreedyExecutionController(
        state,
        planner,
        observer,
        initial_budget_m=60.0,  # TEST ONLY: exactly the executed distance.
    ).execute_next(
        altitude_m=50.0,
        simulated_time_s=3.0,
    )

    assert result.mission_complete
    assert result.plan is None
    assert result.route_execution is None
    assert result.normal_route_replan is None
    assert state.current_position_m == position_before
    assert state.observed_cells == observed_before
    assert state.total_executed_distance_m == pytest.approx(60.0)
    assert observer.anomaly_confirmed


def test_radiation_step_cannot_overspend_positive_remaining_budget():
    probability_map = np.full((5, 5), 0.01)
    probability_map[2, 3] = 0.02
    state, normal_route, _, observer, planner = _pipeline(probability_map)
    for test_only_time_s in (0.0, 1.0, 2.0):
        execute_normal_route_node_with_radiation(
            normal_route,
            observer,
            altitude_m=50.0,
            simulated_time_s=test_only_time_s,
        )
    assert state.total_executed_distance_m == pytest.approx(60.0)

    result = RadiationGreedyExecutionController(
        state,
        planner,
        observer,
        initial_budget_m=80.0,  # TEST ONLY: 20 m cannot fund one 30 m cell step.
    ).execute_next(
        altitude_m=50.0,
        simulated_time_s=3.0,
    )

    assert not result.mission_complete
    assert result.plan.selected_candidate is not None
    assert result.selected_candidate is None
    assert result.route_execution is None
    assert result.normal_route_replan is not None
    assert state.current_position_m == (75.0, 75.0)
    assert state.total_executed_distance_m == pytest.approx(60.0)
