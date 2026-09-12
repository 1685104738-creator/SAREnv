import pytest
from shapely.geometry import LineString

from sarenv.analytics.route_execution import (
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
    execute_normal_route_node_with_radiation,
)
from sarenv.radiation.common.grid import GridSpec
from sarenv.radiation.online.estimator import IncrementalRadiationGrid
from sarenv.radiation.online.sensor import NoiseFreeRadiationSensor
from sarenv.radiation.online.trigger import (
    AboveBackgroundCriterion,
    ConsecutiveRadiationTrigger,
)


BOUNDS = (0.0, 0.0, 90.0, 90.0)


def _state() -> ExecutedSARState:
    return ExecutedSARState(
        map_shape=(3, 3),
        bounds=BOUNDS,
        detection_radius_m=0.0,
    )


def test_precomputed_future_nodes_do_not_enter_executed_sar_state():
    state = _state()
    controller = PrecomputedRouteController(
        LineString([(15.0, 15.0), (45.0, 15.0), (75.0, 15.0)]),
        state,
    )

    assert state.current_position_m is None
    assert state.observed_cells == frozenset()
    assert state.executed_positions_m == ()

    first = controller.execute_next_node()

    assert first.node_index == 0
    assert first.distance_from_previous_m == 0.0
    assert state.current_position_m == (15.0, 15.0)
    assert state.total_executed_distance_m == 0.0
    assert state.current_grid_position == (0, 0)
    assert state.observed_cells == frozenset({(0, 0)})
    assert (0, 1) not in state.observed_cells
    assert (0, 2) not in state.observed_cells


def test_each_call_executes_exactly_one_node_and_updates_coverage_after_arrival():
    state = _state()
    controller = PrecomputedRouteController(
        LineString([(15.0, 15.0), (45.0, 15.0), (75.0, 15.0)]),
        state,
    )

    controller.execute_next_node()
    second = controller.execute_next_node()

    assert second.node_index == 1
    assert second.distance_from_previous_m == pytest.approx(30.0)
    assert state.current_position_m == (45.0, 15.0)
    assert state.observed_cells == frozenset({(0, 0), (0, 1)})
    assert state.executed_positions_m == ((15.0, 15.0), (45.0, 15.0))
    assert state.total_executed_distance_m == pytest.approx(30.0)
    assert state.remaining_budget_m(100.0) == pytest.approx(70.0)
    assert controller.next_node_index == 2


def test_invalidating_route_discards_only_unexecuted_future_nodes():
    state = _state()
    controller = PrecomputedRouteController(
        LineString([(15.0, 15.0), (45.0, 15.0), (75.0, 15.0)]),
        state,
    )
    controller.execute_next_node()
    controller.execute_next_node()

    discarded = controller.invalidate_remaining()

    assert discarded == 1
    assert controller.invalidated
    assert not controller.has_next
    assert state.current_position_m == (45.0, 15.0)
    assert state.observed_cells == frozenset({(0, 0), (0, 1)})
    assert state.total_executed_distance_m == pytest.approx(30.0)
    assert (0, 2) not in state.observed_cells
    with pytest.raises(RuntimeError, match="invalidated"):
        controller.execute_next_node()


def test_temporary_node_sampling_interrupts_only_after_third_executed_anomaly():
    bounds = (0.0, 0.0, 120.0, 30.0)
    state = ExecutedSARState(
        map_shape=(1, 4),
        bounds=bounds,
        detection_radius_m=0.0,
    )
    controller = PrecomputedRouteController(
        LineString(
            [(15.0, 15.0), (45.0, 15.0), (75.0, 15.0), (105.0, 15.0)]
        ),
        state,
    )
    quantity = "test_radiation_rate"
    unit = "test_unit"
    queried_positions = []

    def test_only_truth(x_m, y_m, altitude_m):
        queried_positions.append((x_m, y_m, altitude_m))
        return 2.0

    sensor = NoiseFreeRadiationSensor(
        test_only_truth,
        quantity=quantity,
        unit=unit,
    )
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(bounds, "EPSG:32630"),
        quantity=quantity,
        unit=unit,
    )
    trigger = ConsecutiveRadiationTrigger(
        AboveBackgroundCriterion(1.0, quantity=quantity, unit=unit)
    )
    observer = ExecutedNodeRadiationObserver(sensor, estimator, trigger)

    first = execute_normal_route_node_with_radiation(
        controller,
        observer,
        altitude_m=50.0,
        simulated_time_s=0.0,
    )
    second = execute_normal_route_node_with_radiation(
        controller,
        observer,
        altitude_m=50.0,
        simulated_time_s=1.0,
    )

    assert not first.normal_route_interrupted
    assert not second.normal_route_interrupted
    assert controller.has_next
    assert not controller.invalidated

    third = execute_normal_route_node_with_radiation(
        controller,
        observer,
        altitude_m=50.0,
        simulated_time_s=2.0,
    )

    assert third.normal_route_interrupted
    assert third.radiation_observation.anomaly_confirmed
    assert controller.invalidated
    assert controller.discarded_node_count == 1
    assert state.current_position_m == (75.0, 15.0)
    assert state.executed_positions_m == (
        (15.0, 15.0),
        (45.0, 15.0),
        (75.0, 15.0),
    )
    assert state.observed_cells == frozenset({(0, 0), (0, 1), (0, 2)})
    assert (0, 3) not in state.observed_cells
    assert queried_positions == [
        (15.0, 15.0, 50.0),
        (45.0, 15.0, 50.0),
        (75.0, 15.0, 50.0),
    ]


def test_unconfirmed_radiation_never_interrupts_precomputed_normal_route():
    bounds = (0.0, 0.0, 90.0, 30.0)
    state = ExecutedSARState(
        map_shape=(1, 3),
        bounds=bounds,
        detection_radius_m=0.0,
    )
    route = LineString([(15.0, 15.0), (45.0, 15.0), (75.0, 15.0)])
    controller = PrecomputedRouteController(route, state)
    quantity = "test_radiation_rate"
    unit = "test_unit"
    sensor = NoiseFreeRadiationSensor(
        lambda x_m, y_m, altitude_m: 0.5,
        quantity=quantity,
        unit=unit,
    )
    estimator = IncrementalRadiationGrid(
        GridSpec.from_bounds(bounds, "EPSG:32630"),
        quantity=quantity,
        unit=unit,
    )
    trigger = ConsecutiveRadiationTrigger(
        AboveBackgroundCriterion(1.0, quantity=quantity, unit=unit)
    )
    observer = ExecutedNodeRadiationObserver(sensor, estimator, trigger)

    results = [
        execute_normal_route_node_with_radiation(
            controller,
            observer,
            altitude_m=50.0,
            simulated_time_s=test_only_time_s,
        )
        for test_only_time_s in (0.0, 1.0, 2.0)
    ]

    assert all(not result.normal_route_interrupted for result in results)
    assert not controller.invalidated
    assert not controller.has_next
    assert state.executed_positions_m == tuple(route.coords)
    assert state.observed_cells == frozenset({(0, 0), (0, 1), (0, 2)})
