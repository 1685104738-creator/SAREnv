"""Minimal node-level execution state for precomputed SAR routes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
from shapely.geometry import LineString

from sarenv.radiation.online.interfaces import (
    OnlineRadiationEstimator,
    RadiationMeasurementSource,
)
from sarenv.radiation.online.measurement import RadiationMeasurement

from .paths import (
    generate_greedy_continuation_route,
    greedy_visible_cells,
    greedy_world_to_grid_position,
)
from .radiation_priority import (
    GreedyCandidate,
    RadiationGreedyStepPlan,
    RadiationGreedyStepPlanner,
)


@dataclass(frozen=True)
class RouteNodeExecution:
    """One route node that has actually been reached by the controller."""

    node_index: int
    position_m: tuple[float, float]
    grid_position: tuple[int, int]
    distance_from_previous_m: float


@dataclass(frozen=True)
class ExecutedNodeRadiationObservation:
    """Radiation result produced only after a route node has been reached."""

    measurement: RadiationMeasurement
    updated_radiation_cells: int
    anomaly_confirmed: bool
    hazard_transition: str | None = None
    estimator_update_time_s: float = 0.0
    hazard_episode_index: int | None = None
    entry_multiplier_initial: float | None = None
    entry_threshold_current: float | None = None
    exit_threshold_current: float | None = None
    base_hazard_reference: float | None = None


@dataclass(frozen=True)
class NormalRouteNodeResult:
    """One executed normal-route node and its optional interruption event."""

    route_execution: RouteNodeExecution
    radiation_observation: ExecutedNodeRadiationObservation
    normal_route_interrupted: bool


@dataclass(frozen=True)
class NormalRouteReplanRequest:
    """Executed SAR state required to generate a new normal route safely."""

    current_position_m: tuple[float, float]
    current_grid_position: tuple[int, int]
    observed_cells: frozenset[tuple[int, int]]
    reason: str = "radiation_planner_requested_normal"


@dataclass(frozen=True)
class RadiationModeNodeResult:
    """One radiation-mode plan, with either one execution or a replan request."""

    plan: RadiationGreedyStepPlan | None
    selected_candidate: GreedyCandidate | None
    route_execution: RouteNodeExecution | None
    radiation_observation: ExecutedNodeRadiationObservation | None
    normal_route_replan: NormalRouteReplanRequest | None
    mission_complete: bool


@dataclass(frozen=True)
class NormalContinuationResult:
    """A new normal route, or a mission-end result when budget is exhausted."""

    remaining_budget_m: float | None
    remaining_steps: int | None
    route: LineString | None
    controller: PrecomputedRouteController | None
    mission_complete: bool
    completion_reason: str | None = None


class ExecutedSARState:
    """Maintain SAR coverage from reached route nodes only."""

    def __init__(
        self,
        *,
        map_shape: tuple[int, int],
        bounds: tuple[float, float, float, float],
        detection_radius_m: float,
    ) -> None:
        height, width = map_shape
        if height <= 0 or width <= 0:
            raise ValueError("map_shape dimensions must be positive.")
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("bounds must contain four finite values.")
        minx, miny, maxx, maxy = bounds
        if maxx <= minx or maxy <= miny:
            raise ValueError("bounds must have positive width and height.")
        if not math.isfinite(detection_radius_m) or detection_radius_m < 0.0:
            raise ValueError("detection_radius_m must be finite and non-negative.")

        self.map_shape = (int(height), int(width))
        self.bounds = tuple(float(value) for value in bounds)
        self.detection_radius_m = float(detection_radius_m)
        self._current_position_m: tuple[float, float] | None = None
        self._current_grid_position: tuple[int, int] | None = None
        self._observed_cells: set[tuple[int, int]] = set()
        self._executed_positions_m: list[tuple[float, float]] = []
        self._total_executed_distance_m = 0.0

    @property
    def current_position_m(self) -> tuple[float, float] | None:
        return self._current_position_m

    @property
    def current_grid_position(self) -> tuple[int, int] | None:
        return self._current_grid_position

    @property
    def observed_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._observed_cells)

    @property
    def executed_positions_m(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._executed_positions_m)

    @property
    def total_executed_distance_m(self) -> float:
        """Return distance between actually reached positions only."""
        return self._total_executed_distance_m

    @property
    def executed_node_count(self) -> int:
        """Return the number of actually reached route nodes across all modes."""
        return len(self._executed_positions_m)

    @property
    def executed_move_count(self) -> int:
        """Return completed inter-node moves; the initial node costs no move."""
        return max(0, len(self._executed_positions_m) - 1)

    def remaining_budget_m(self, initial_budget_m: float) -> float:
        """Return the shared distance budget remaining across all modes."""
        if not math.isfinite(initial_budget_m) or initial_budget_m < 0.0:
            raise ValueError("initial_budget_m must be finite and non-negative.")
        return max(0.0, float(initial_budget_m) - self._total_executed_distance_m)

    def record_node_arrival(
        self, x_m: float, y_m: float
    ) -> tuple[int, int]:
        """Record one reached node and add only its camera-visible SAR cells."""
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("Executed node coordinates must be finite.")
        minx, miny, maxx, maxy = self.bounds
        if not (minx <= x_m < maxx and miny <= y_m < maxy):
            raise ValueError("Executed node lies outside the SAR bounds.")

        grid_position = greedy_world_to_grid_position(
            x_m,
            y_m,
            map_shape=self.map_shape,
            bounds=self.bounds,
        )
        visible = greedy_visible_cells(
            *grid_position,
            map_shape=self.map_shape,
            bounds=self.bounds,
            detection_radius_m=self.detection_radius_m,
        )
        position = (float(x_m), float(y_m))
        if self._current_position_m is not None:
            self._total_executed_distance_m += math.dist(
                self._current_position_m,
                position,
            )
        self._current_position_m = position
        self._current_grid_position = grid_position
        self._observed_cells.update(visible)
        self._executed_positions_m.append(position)
        return grid_position


class PrecomputedRouteController:
    """Execute a precomputed route one existing LineString node at a time."""

    def __init__(
        self,
        route: LineString,
        state: ExecutedSARState,
        *,
        first_node_already_executed: bool = False,
    ) -> None:
        if not isinstance(route, LineString):
            raise TypeError("route must be a LineString.")
        if not isinstance(state, ExecutedSARState):
            raise TypeError("state must be an ExecutedSARState.")
        self.route = route
        self.state = state
        self._route_nodes = tuple(
            (float(x_m), float(y_m)) for x_m, y_m in route.coords
        )
        if first_node_already_executed and self._route_nodes:
            if state.current_position_m != self._route_nodes[0]:
                raise ValueError(
                    "An already-executed first node must match current position."
                )
            self._next_node_index = 1
        else:
            self._next_node_index = 0
        self._invalidated = False
        self._discarded_node_count = 0

    @property
    def has_next(self) -> bool:
        return not self._invalidated and self._next_node_index < len(
            self._route_nodes
        )

    @property
    def invalidated(self) -> bool:
        return self._invalidated

    @property
    def next_node_index(self) -> int:
        return self._next_node_index

    @property
    def discarded_node_count(self) -> int:
        return self._discarded_node_count

    def execute_next_node(self) -> RouteNodeExecution | None:
        """Reach exactly one route node; return ``None`` after normal exhaustion."""
        if self._invalidated:
            raise RuntimeError("The remaining precomputed route has been invalidated.")
        if not self.has_next:
            return None

        node_index = self._next_node_index
        position = self._route_nodes[node_index]
        previous = self.state.current_position_m
        distance_m = 0.0 if previous is None else math.dist(previous, position)
        grid_position = self.state.record_node_arrival(*position)
        self._next_node_index += 1
        return RouteNodeExecution(
            node_index=node_index,
            position_m=position,
            grid_position=grid_position,
            distance_from_previous_m=distance_m,
        )

    def invalidate_remaining(self) -> int:
        """Stop this route permanently and return the number of discarded nodes."""
        if self._invalidated:
            return self._discarded_node_count
        self._discarded_node_count = len(self._route_nodes) - self._next_node_index
        self._invalidated = True
        return self._discarded_node_count


class ExecutedNodeRadiationObserver:
    """Temporary structural node-sampling adapter, not an experiment schedule."""

    def __init__(
        self,
        measurement_source: RadiationMeasurementSource,
        estimator: OnlineRadiationEstimator,
        trigger,
    ) -> None:
        if not callable(getattr(measurement_source, "measure", None)):
            raise TypeError("measurement_source must provide measure().")
        if not callable(getattr(estimator, "update", None)):
            raise TypeError("estimator must provide update().")
        required_trigger_attributes = (
            "observe",
            "reset",
            "confirmed",
            "consecutive_count",
            "last_transition",
        )
        if not all(hasattr(trigger, name) for name in required_trigger_attributes):
            raise TypeError("trigger does not provide the radiation-mode contract.")
        self._measurement_source = measurement_source
        self._estimator = estimator
        self._trigger = trigger

    @property
    def anomaly_confirmed(self) -> bool:
        return self._trigger.confirmed

    @property
    def consecutive_anomaly_count(self) -> int:
        return self._trigger.consecutive_count

    def reset_after_radiation_episode(self) -> None:
        """Reset only when one radiation-priority episode returns to normal."""
        reset_after_episode = getattr(self._trigger, "reset_after_episode", None)
        if callable(reset_after_episode):
            reset_after_episode()
        else:
            self._trigger.reset()

    def observe_executed_node(
        self,
        execution: RouteNodeExecution,
        *,
        altitude_m: float,
        simulated_time_s: float,
    ) -> ExecutedNodeRadiationObservation:
        """Measure and update only at the node contained in ``execution``."""
        x_m, y_m = execution.position_m
        measurement = self._measurement_source.measure(
            x_m,
            y_m,
            altitude_m,
            simulated_time_s,
        )
        update_started = time.perf_counter()
        updated_cells = self._estimator.update(measurement)
        estimator_update_time_s = time.perf_counter() - update_started
        confirmed = self._trigger.observe(measurement)
        return ExecutedNodeRadiationObservation(
            measurement=measurement,
            updated_radiation_cells=updated_cells,
            anomaly_confirmed=confirmed,
            hazard_transition=self._trigger.last_transition,
            estimator_update_time_s=estimator_update_time_s,
            hazard_episode_index=getattr(
                self._trigger,
                "last_observation_episode_index",
                None,
            ),
            entry_multiplier_initial=getattr(
                self._trigger,
                "initial_entry_multiplier",
                None,
            ),
            entry_threshold_current=getattr(
                self._trigger,
                "last_observation_entry_threshold",
                getattr(self._trigger, "enter_threshold", None),
            ),
            exit_threshold_current=getattr(
                self._trigger,
                "last_observation_exit_threshold",
                getattr(self._trigger, "exit_threshold", None),
            ),
            base_hazard_reference=getattr(
                self._trigger,
                "base_hazard_reference",
                None,
            ),
        )


def execute_normal_route_node_with_radiation(
    controller: PrecomputedRouteController,
    observer: ExecutedNodeRadiationObserver,
    *,
    altitude_m: float,
    simulated_time_s: float,
) -> NormalRouteNodeResult | None:
    """Execute one normal node, observe there, and interrupt on confirmation."""
    execution = controller.execute_next_node()
    if execution is None:
        return None
    observation = observer.observe_executed_node(
        execution,
        altitude_m=altitude_m,
        simulated_time_s=simulated_time_s,
    )
    if observation.anomaly_confirmed:
        controller.invalidate_remaining()
    return NormalRouteNodeResult(
        route_execution=execution,
        radiation_observation=observation,
        normal_route_interrupted=observation.anomaly_confirmed,
    )


class RadiationGreedyExecutionController:
    """Plan and execute at most one local radiation-priority step per call."""

    def __init__(
        self,
        state: ExecutedSARState,
        planner: RadiationGreedyStepPlanner,
        observer: ExecutedNodeRadiationObserver,
        *,
        initial_budget_m: float | None = None,
        mission_step_allowance: int | None = None,
    ) -> None:
        if not isinstance(state, ExecutedSARState):
            raise TypeError("state must be an ExecutedSARState.")
        if not isinstance(planner, RadiationGreedyStepPlanner):
            raise TypeError("planner must be a RadiationGreedyStepPlanner.")
        if not isinstance(observer, ExecutedNodeRadiationObserver):
            raise TypeError("observer must be an ExecutedNodeRadiationObserver.")
        if initial_budget_m is None and mission_step_allowance is None:
            raise ValueError("A distance budget or mission step allowance is required.")
        if initial_budget_m is not None and (
            not math.isfinite(initial_budget_m) or initial_budget_m < 0.0
        ):
            raise ValueError("initial_budget_m must be finite and non-negative.")
        if mission_step_allowance is not None and (
            isinstance(mission_step_allowance, bool)
            or not isinstance(mission_step_allowance, int)
            or mission_step_allowance <= 0
        ):
            raise ValueError("mission_step_allowance must be a positive integer.")
        self.state = state
        self.planner = planner
        self.observer = observer
        self.initial_budget_m = (
            None if initial_budget_m is None else float(initial_budget_m)
        )
        self.mission_step_allowance = mission_step_allowance
        self._step_index = 0
        self._normal_replan_requested = False

    def execute_next(
        self,
        *,
        altitude_m: float,
        simulated_time_s: float,
    ) -> RadiationModeNodeResult:
        """Plan from current executed state and move to at most one neighbour."""
        if self._normal_replan_requested:
            raise RuntimeError("A new normal route has already been requested.")
        if not self.observer.anomaly_confirmed:
            raise RuntimeError("Radiation Greedy requires a confirmed anomaly.")
        if self.state.current_grid_position is None:
            raise RuntimeError("Radiation Greedy requires an executed current position.")
        if self._mission_limit_reached():
            return RadiationModeNodeResult(
                plan=None,
                selected_candidate=None,
                route_execution=None,
                radiation_observation=None,
                normal_route_replan=None,
                mission_complete=True,
            )

        plan = self.planner.plan_next(
            self.state.current_grid_position,
            self.state.observed_cells,
            radiation_priority_enabled=True,
        )
        candidate = plan.selected_candidate
        if candidate is None:
            self._normal_replan_requested = True
            return RadiationModeNodeResult(
                plan=plan,
                selected_candidate=None,
                route_execution=None,
                radiation_observation=None,
                normal_route_replan=self._normal_replan_request(
                    "no_valid_radiation_candidate"
                ),
                mission_complete=False,
            )

        previous_position = self.state.current_position_m
        position = (candidate.x_m, candidate.y_m)
        distance_m = math.dist(previous_position, position)
        if self.initial_budget_m is not None and distance_m > self.state.remaining_budget_m(
            self.initial_budget_m
        ):
            self._normal_replan_requested = True
            return RadiationModeNodeResult(
                plan=plan,
                selected_candidate=None,
                route_execution=None,
                radiation_observation=None,
                normal_route_replan=self._normal_replan_request(
                    "insufficient_distance_budget"
                ),
                mission_complete=False,
            )
        grid_position = self.state.record_node_arrival(*position)
        if grid_position != (candidate.row, candidate.col):
            raise RuntimeError("Selected candidate does not match the executed SAR cell.")
        execution = RouteNodeExecution(
            node_index=self._step_index,
            position_m=position,
            grid_position=grid_position,
            distance_from_previous_m=distance_m,
        )
        self._step_index += 1
        observation = self.observer.observe_executed_node(
            execution,
            altitude_m=altitude_m,
            simulated_time_s=simulated_time_s,
        )
        normal_route_replan = None
        if observation.hazard_transition == "exit":
            self._normal_replan_requested = True
            normal_route_replan = self._normal_replan_request(
                "hazard_exit_threshold"
            )
        return RadiationModeNodeResult(
            plan=plan,
            selected_candidate=candidate,
            route_execution=execution,
            radiation_observation=observation,
            normal_route_replan=normal_route_replan,
            mission_complete=False,
        )

    def _mission_limit_reached(self) -> bool:
        if self.mission_step_allowance is not None and (
            self.state.executed_move_count >= self.mission_step_allowance
        ):
            return True
        return self.initial_budget_m is not None and (
            self.state.remaining_budget_m(self.initial_budget_m) <= 0.0
        )

    def _normal_replan_request(self, reason: str) -> NormalRouteReplanRequest:
        if self.state.current_position_m is None:
            raise RuntimeError("Normal replanning requires a current position.")
        if self.state.current_grid_position is None:
            raise RuntimeError("Normal replanning requires a current grid cell.")
        return NormalRouteReplanRequest(
            current_position_m=self.state.current_position_m,
            current_grid_position=self.state.current_grid_position,
            observed_cells=self.state.observed_cells,
            reason=reason,
        )


def resume_normal_route_after_radiation(
    request: NormalRouteReplanRequest,
    state: ExecutedSARState,
    observer: ExecutedNodeRadiationObserver,
    *,
    initial_budget_m: float | None = None,
    mission_step_allowance: int | None = None,
    rng: np.random.Generator | None = None,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    search_center: tuple[float, float],
    max_radius: float,
    fov_deg: float,
    altitude: float,
) -> NormalContinuationResult:
    """Reset the episode trigger and create a state-aware normal continuation."""
    if not isinstance(request, NormalRouteReplanRequest):
        raise TypeError("request must be a NormalRouteReplanRequest.")
    if request.current_position_m != state.current_position_m:
        raise ValueError("Replan request current position is stale.")
    if request.current_grid_position != state.current_grid_position:
        raise ValueError("Replan request current SAR cell is stale.")
    if request.observed_cells != state.observed_cells:
        raise ValueError("Replan request observed SAR state is stale.")

    if initial_budget_m is None and mission_step_allowance is None:
        raise ValueError("A distance budget or mission step allowance is required.")
    remaining_budget_m = (
        None
        if initial_budget_m is None
        else state.remaining_budget_m(initial_budget_m)
    )
    remaining_steps = (
        None
        if mission_step_allowance is None
        else max(0, mission_step_allowance - state.executed_move_count)
    )
    if (remaining_budget_m is not None and remaining_budget_m <= 0.0) or (
        remaining_steps is not None and remaining_steps <= 0
    ):
        return NormalContinuationResult(
            remaining_budget_m=(
                0.0 if remaining_budget_m is not None else None
            ),
            remaining_steps=(0 if remaining_steps is not None else None),
            route=None,
            controller=None,
            mission_complete=True,
            completion_reason="mission_limit_reached",
        )

    route = generate_greedy_continuation_route(
        current_position_m=request.current_position_m,
        observed_cells=request.observed_cells,
        search_center=search_center,
        remaining_budget_m=remaining_budget_m,
        remaining_steps=remaining_steps,
        probability_map=probability_map,
        bounds=bounds,
        max_radius=max_radius,
        fov_deg=fov_deg,
        altitude=altitude,
        rng=rng,
    )
    controller = PrecomputedRouteController(
        route,
        state,
        first_node_already_executed=True,
    )
    observer.reset_after_radiation_episode()
    return NormalContinuationResult(
        remaining_budget_m=remaining_budget_m,
        remaining_steps=remaining_steps,
        route=route,
        controller=controller,
        mission_complete=False,
    )


__all__ = [
    "ExecutedSARState",
    "ExecutedNodeRadiationObservation",
    "ExecutedNodeRadiationObserver",
    "NormalRouteNodeResult",
    "NormalContinuationResult",
    "NormalRouteReplanRequest",
    "PrecomputedRouteController",
    "RadiationGreedyExecutionController",
    "RadiationModeNodeResult",
    "RouteNodeExecution",
    "execute_normal_route_node_with_radiation",
    "resume_normal_route_after_radiation",
]
