import numpy as np
import pytest
from shapely.geometry import LineString

from sarenv.analytics.mission_mode import MissionMode, MissionModeController
from sarenv.analytics.radiation_priority import RadiationGreedyStepPlanner
from sarenv.analytics.route_execution import (
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
)
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.sensor import NoiseFreeRadiationSensor
from sarenv.radiation.online.trigger import (
    AboveBackgroundCriterion,
    ConsecutiveRadiationTrigger,
)


BOUNDS = (0.0, 0.0, 150.0, 150.0)
QUANTITY = "test_radiation_rate"
UNIT = "test_unit"


def _controller(
    probability_map: np.ndarray,
    *,
    measurement_value: float,
    initial_budget_m: float = 300.0,
) -> tuple[MissionModeController, ExecutedSARState]:
    state = ExecutedSARState(
        map_shape=probability_map.shape,
        bounds=BOUNDS,
        detection_radius_m=0.0,
    )
    normal_controller = PrecomputedRouteController(
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
    sensor = NoiseFreeRadiationSensor(
        lambda x_m, y_m, altitude_m: measurement_value,
        quantity=QUANTITY,
        unit=UNIT,
    )
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(BOUNDS, "EPSG:32630"),
        quantity=QUANTITY,
        unit=UNIT,
    )
    trigger = ConsecutiveRadiationTrigger(
        AboveBackgroundCriterion(1.0, quantity=QUANTITY, unit=UNIT)
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
    controller = MissionModeController(
        state=state,
        normal_route_controller=normal_controller,
        observer=observer,
        radiation_planner=planner,
        initial_budget_m=initial_budget_m,
        probability_map=probability_map,
        bounds=BOUNDS,
        search_center=(75.0, 75.0),
        max_radius=200.0,
        fov_deg=0.0,
        altitude_m=50.0,
    )
    return controller, state


def test_normal_stays_normal_without_confirmed_radiation():
    controller, state = _controller(
        np.full((5, 5), 0.01),
        measurement_value=0.5,
    )

    result = controller.advance(simulated_time_s=0.0)

    assert result.mode_before is MissionMode.NORMAL
    assert result.mode_after is MissionMode.NORMAL
    assert result.moved
    assert state.current_position_m == (15.0, 75.0)


def test_third_anomaly_changes_normal_to_radiation():
    controller, _ = _controller(
        np.full((5, 5), 0.01),
        measurement_value=2.0,
    )

    results = [controller.advance(simulated_time_s=0.0) for _ in range(3)]

    assert [result.mode_after for result in results] == [
        MissionMode.NORMAL,
        MissionMode.NORMAL,
        MissionMode.RADIATION,
    ]
    assert results[-1].normal_route_interrupted


def test_radiation_executes_one_step_and_remains_radiation():
    probability_map = np.full((5, 5), 0.01)
    probability_map[2, 3] = 0.02
    controller, state = _controller(probability_map, measurement_value=2.0)
    for _ in range(3):
        controller.advance(simulated_time_s=0.0)

    result = controller.advance(simulated_time_s=0.0)

    assert result.mode_before is MissionMode.RADIATION
    assert result.mode_after is MissionMode.RADIATION
    assert result.moved
    assert result.route_execution.distance_from_previous_m == pytest.approx(30.0)
    assert state.current_position_m == (105.0, 75.0)


def test_radiation_exit_replans_normal_and_later_retriggers():
    probability_map = np.full((5, 5), 0.0009)
    probability_map[2, 3] = 0.02
    controller, state = _controller(probability_map, measurement_value=2.0)
    for _ in range(3):
        controller.advance(simulated_time_s=0.0)

    radiation_move = controller.advance(simulated_time_s=0.0)
    radiation_exit = controller.advance(simulated_time_s=0.0)

    assert radiation_move.mode_after is MissionMode.RADIATION
    assert not radiation_exit.moved
    assert radiation_exit.normal_route_replanned
    assert radiation_exit.mode_after is MissionMode.NORMAL
    assert controller.remaining_budget_m == pytest.approx(210.0)

    later = [controller.advance(simulated_time_s=0.0) for _ in range(3)]

    assert [result.mode_after for result in later] == [
        MissionMode.NORMAL,
        MissionMode.NORMAL,
        MissionMode.RADIATION,
    ]
    later_distance_m = sum(
        result.route_execution.distance_from_previous_m for result in later
    )
    assert state.total_executed_distance_m == pytest.approx(
        90.0 + later_distance_m
    )
    assert controller.mode_history == (
        MissionMode.NORMAL,
        MissionMode.RADIATION,
        MissionMode.NORMAL,
        MissionMode.RADIATION,
    )


def test_budget_exhaustion_completes_and_complete_never_moves_again():
    controller, state = _controller(
        np.full((5, 5), 0.01),
        measurement_value=0.5,
        initial_budget_m=30.0,
    )

    first = controller.advance(simulated_time_s=0.0)
    second = controller.advance(simulated_time_s=0.0)
    positions_at_completion = state.executed_positions_m
    after_complete = controller.advance(simulated_time_s=0.0)

    assert first.mode_after is MissionMode.NORMAL
    assert second.mode_after is MissionMode.COMPLETE
    assert second.mission_complete
    assert state.total_executed_distance_m == pytest.approx(30.0)
    assert not after_complete.moved
    assert after_complete.mode_before is MissionMode.COMPLETE
    assert after_complete.mode_after is MissionMode.COMPLETE
    assert state.executed_positions_m == positions_at_completion


def test_radiation_step_exhausting_budget_changes_directly_to_complete():
    probability_map = np.full((5, 5), 0.01)
    probability_map[2, 3] = 0.02
    controller, state = _controller(
        probability_map,
        measurement_value=2.0,
        initial_budget_m=90.0,
    )
    for _ in range(3):
        controller.advance(simulated_time_s=0.0)

    result = controller.advance(simulated_time_s=0.0)

    assert result.mode_before is MissionMode.RADIATION
    assert result.mode_after is MissionMode.COMPLETE
    assert result.mission_complete
    assert result.moved
    assert state.total_executed_distance_m == pytest.approx(90.0)
