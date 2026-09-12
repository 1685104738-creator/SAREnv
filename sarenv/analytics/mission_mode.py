"""Thin orchestration of the existing normal and radiation execution layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time

import numpy as np

from .paths import greedy_searchable_cells
from .radiation_priority import RadiationGreedyStepPlan, RadiationGreedyStepPlanner
from .route_execution import (
    ExecutedNodeRadiationObservation,
    ExecutedNodeRadiationObserver,
    ExecutedSARState,
    PrecomputedRouteController,
    RadiationGreedyExecutionController,
    RouteNodeExecution,
    execute_normal_route_node_with_radiation,
    resume_normal_route_after_radiation,
)


class MissionMode(str, Enum):
    """The only execution modes in the v1 structural mission scaffold."""

    NORMAL = "NORMAL"
    RADIATION = "RADIATION"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class MissionModeStepResult:
    """One controller call, which may execute one node or only change mode."""

    controller_call_index: int
    executed_step_index: int | None
    mode_before: MissionMode
    mode_after: MissionMode
    route_execution: RouteNodeExecution | None
    radiation_observation: ExecutedNodeRadiationObservation | None
    radiation_plan: RadiationGreedyStepPlan | None
    normal_route_interrupted: bool
    normal_route_replanned: bool
    continuation_planning_time_s: float | None
    mission_complete: bool
    mode_transition_reason: str | None = None
    completion_reason: str | None = None

    @property
    def moved(self) -> bool:
        """Return whether this call reached an actual route node."""
        return self.route_execution is not None


class MissionModeController:
    """Delegate NORMAL/RADIATION/COMPLETE transitions to existing components."""

    def __init__(
        self,
        *,
        state: ExecutedSARState,
        normal_route_controller: PrecomputedRouteController,
        observer: ExecutedNodeRadiationObserver,
        radiation_planner: RadiationGreedyStepPlanner,
        initial_budget_m: float | None,
        probability_map: np.ndarray,
        bounds: tuple[float, float, float, float],
        search_center: tuple[float, float],
        max_radius: float,
        fov_deg: float,
        altitude_m: float,
        mission_step_allowance: int | None = None,
        planner_rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(state, ExecutedSARState):
            raise TypeError("state must be an ExecutedSARState.")
        if not isinstance(normal_route_controller, PrecomputedRouteController):
            raise TypeError(
                "normal_route_controller must be a PrecomputedRouteController."
            )
        if normal_route_controller.state is not state:
            raise ValueError("Normal route controller must share the executed state.")
        if not isinstance(observer, ExecutedNodeRadiationObserver):
            raise TypeError("observer must be an ExecutedNodeRadiationObserver.")
        if not isinstance(radiation_planner, RadiationGreedyStepPlanner):
            raise TypeError("radiation_planner must be a RadiationGreedyStepPlanner.")
        numeric = {
            "max_radius": max_radius,
            "fov_deg": fov_deg,
            "altitude_m": altitude_m,
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("Mission configuration values must be finite.")
        if initial_budget_m is None and mission_step_allowance is None:
            raise ValueError("A distance budget or mission step allowance is required.")
        if initial_budget_m is not None and (
            not math.isfinite(initial_budget_m) or initial_budget_m < 0.0
        ):
            raise ValueError("initial_budget_m must be non-negative.")
        if mission_step_allowance is not None and (
            isinstance(mission_step_allowance, bool)
            or not isinstance(mission_step_allowance, int)
            or mission_step_allowance <= 0
        ):
            raise ValueError("mission_step_allowance must be a positive integer.")
        if planner_rng is None:
            planner_rng = np.random.default_rng()
        if not isinstance(planner_rng, np.random.Generator):
            raise TypeError("planner_rng must be a numpy.random.Generator.")
        if max_radius <= 0.0:
            raise ValueError("max_radius must be positive.")
        if fov_deg < 0.0 or altitude_m < 0.0:
            raise ValueError("fov_deg and altitude_m must be non-negative.")

        self.state = state
        self.normal_route_controller = normal_route_controller
        self.observer = observer
        self.radiation_planner = radiation_planner
        self.initial_budget_m = (
            None if initial_budget_m is None else float(initial_budget_m)
        )
        self.mission_step_allowance = mission_step_allowance
        self.planner_rng = planner_rng
        self.probability_map = np.asarray(probability_map)
        self.bounds = tuple(float(value) for value in bounds)
        self.search_center = tuple(float(value) for value in search_center)
        self.max_radius = float(max_radius)
        self.fov_deg = float(fov_deg)
        self.altitude_m = float(altitude_m)
        self.searchable_cells = frozenset(
            greedy_searchable_cells(
                map_shape=self.probability_map.shape,
                bounds=self.bounds,
                search_center=self.search_center,
                max_radius=self.max_radius,
            )
        )

        self._mode = MissionMode.NORMAL
        self._mode_history = [self._mode]
        self._radiation_controller: RadiationGreedyExecutionController | None = None
        self._controller_call_index = 0
        self._executed_step_count = 0
        self._last_transition_reason: str | None = None
        self._completion_reason: str | None = None

    @property
    def mode(self) -> MissionMode:
        return self._mode

    @property
    def mode_history(self) -> tuple[MissionMode, ...]:
        """Return state transitions, including the initial NORMAL state."""
        return tuple(self._mode_history)

    @property
    def is_complete(self) -> bool:
        return self._mode is MissionMode.COMPLETE

    @property
    def executed_step_count(self) -> int:
        return self._executed_step_count

    @property
    def remaining_budget_m(self) -> float | None:
        if self.initial_budget_m is None:
            return None
        return self.state.remaining_budget_m(self.initial_budget_m)

    @property
    def remaining_steps(self) -> int | None:
        if self.mission_step_allowance is None:
            return None
        return max(0, self.mission_step_allowance - self.state.executed_move_count)

    @property
    def completion_reason(self) -> str | None:
        return self._completion_reason

    def advance(self, *, simulated_time_s: float) -> MissionModeStepResult:
        """Execute at most one real node and perform only required mode changes."""
        if not math.isfinite(simulated_time_s) or simulated_time_s < 0.0:
            raise ValueError("simulated_time_s must be finite and non-negative.")

        call_index = self._controller_call_index
        self._controller_call_index += 1
        mode_before = self._mode
        self._last_transition_reason = None

        if self.is_complete:
            return self._result(call_index, mode_before)
        if self._mission_limit_reached():
            self._complete("mission_limit_reached")
            return self._result(call_index, mode_before)

        if self._mode is MissionMode.NORMAL:
            return self._advance_normal(call_index, simulated_time_s)
        return self._advance_radiation(call_index, simulated_time_s)

    def _advance_normal(
        self,
        call_index: int,
        simulated_time_s: float,
    ) -> MissionModeStepResult:
        mode_before = self._mode
        normal_result = execute_normal_route_node_with_radiation(
            self.normal_route_controller,
            self.observer,
            altitude_m=self.altitude_m,
            simulated_time_s=simulated_time_s,
        )
        if normal_result is None:
            self._complete("normal_route_complete")
            return self._result(call_index, mode_before)

        executed_step_index = self._record_executed_step()
        if self._coverage_complete():
            self._complete("all_searchable_cells_covered")
        elif self._mission_limit_reached():
            self._complete("mission_limit_reached")
        elif normal_result.normal_route_interrupted:
            self._radiation_controller = RadiationGreedyExecutionController(
                self.state,
                self.radiation_planner,
                self.observer,
                initial_budget_m=self.initial_budget_m,
                mission_step_allowance=self.mission_step_allowance,
            )
            self._transition_to(MissionMode.RADIATION, "hazard_enter_threshold")
        elif not self.normal_route_controller.has_next:
            self._complete("normal_route_complete")

        return self._result(
            call_index,
            mode_before,
            executed_step_index=executed_step_index,
            route_execution=normal_result.route_execution,
            radiation_observation=normal_result.radiation_observation,
            normal_route_interrupted=normal_result.normal_route_interrupted,
        )

    def _advance_radiation(
        self,
        call_index: int,
        simulated_time_s: float,
    ) -> MissionModeStepResult:
        mode_before = self._mode
        if self._radiation_controller is None:
            raise RuntimeError("RADIATION mode requires its execution controller.")

        radiation_result = self._radiation_controller.execute_next(
            altitude_m=self.altitude_m,
            simulated_time_s=simulated_time_s,
        )
        if radiation_result.mission_complete:
            self._complete("mission_limit_reached")
            return self._result(
                call_index,
                mode_before,
                radiation_plan=radiation_result.plan,
            )

        executed_step_index = None
        if radiation_result.route_execution is not None:
            executed_step_index = self._record_executed_step()
            if self._coverage_complete():
                self._complete("all_searchable_cells_covered")
            elif self._mission_limit_reached():
                self._complete("mission_limit_reached")
            if self.is_complete:
                return self._result(
                    call_index,
                    mode_before,
                    executed_step_index=executed_step_index,
                    route_execution=radiation_result.route_execution,
                    radiation_observation=radiation_result.radiation_observation,
                    radiation_plan=radiation_result.plan,
                )

        if radiation_result.normal_route_replan is not None:
            planning_started = time.perf_counter()
            continuation = resume_normal_route_after_radiation(
                radiation_result.normal_route_replan,
                self.state,
                self.observer,
                initial_budget_m=self.initial_budget_m,
                mission_step_allowance=self.mission_step_allowance,
                rng=self.planner_rng,
                probability_map=self.probability_map,
                bounds=self.bounds,
                search_center=self.search_center,
                max_radius=self.max_radius,
                fov_deg=self.fov_deg,
                altitude=self.altitude_m,
            )
            continuation_planning_time_s = time.perf_counter() - planning_started
            self._radiation_controller = None
            if continuation.mission_complete:
                self._complete(
                    continuation.completion_reason or "mission_limit_reached"
                )
            else:
                if continuation.controller is None:
                    raise RuntimeError("Continuation result omitted its controller.")
                self.normal_route_controller = continuation.controller
                if self.normal_route_controller.has_next:
                    self._transition_to(
                        MissionMode.NORMAL,
                        radiation_result.normal_route_replan.reason,
                    )
                else:
                    self._complete("normal_continuation_empty")
            return self._result(
                call_index,
                mode_before,
                executed_step_index=executed_step_index,
                route_execution=radiation_result.route_execution,
                radiation_observation=radiation_result.radiation_observation,
                radiation_plan=radiation_result.plan,
                normal_route_replanned=True,
                continuation_planning_time_s=continuation_planning_time_s,
            )

        if radiation_result.route_execution is None:
            raise RuntimeError("Radiation step returned neither movement nor replan.")
        return self._result(
            call_index,
            mode_before,
            executed_step_index=executed_step_index,
            route_execution=radiation_result.route_execution,
            radiation_observation=radiation_result.radiation_observation,
            radiation_plan=radiation_result.plan,
        )

    def _record_executed_step(self) -> int:
        index = self._executed_step_count
        self._executed_step_count += 1
        return index

    def _mission_limit_reached(self) -> bool:
        if self.remaining_steps is not None and self.remaining_steps <= 0:
            return True
        return self.remaining_budget_m is not None and self.remaining_budget_m <= 0.0

    def _coverage_complete(self) -> bool:
        return self.searchable_cells.issubset(self.state.observed_cells)

    def _complete(self, reason: str) -> None:
        self._completion_reason = reason
        self._transition_to(MissionMode.COMPLETE, reason)

    def _transition_to(self, mode: MissionMode, reason: str | None = None) -> None:
        if mode is self._mode:
            return
        self._mode = mode
        self._mode_history.append(mode)
        self._last_transition_reason = reason

    def _result(
        self,
        call_index: int,
        mode_before: MissionMode,
        *,
        executed_step_index: int | None = None,
        route_execution: RouteNodeExecution | None = None,
        radiation_observation: ExecutedNodeRadiationObservation | None = None,
        radiation_plan: RadiationGreedyStepPlan | None = None,
        normal_route_interrupted: bool = False,
        normal_route_replanned: bool = False,
        continuation_planning_time_s: float | None = None,
    ) -> MissionModeStepResult:
        return MissionModeStepResult(
            controller_call_index=call_index,
            executed_step_index=executed_step_index,
            mode_before=mode_before,
            mode_after=self._mode,
            route_execution=route_execution,
            radiation_observation=radiation_observation,
            radiation_plan=radiation_plan,
            normal_route_interrupted=normal_route_interrupted,
            normal_route_replanned=normal_route_replanned,
            continuation_planning_time_s=continuation_planning_time_s,
            mission_complete=self.is_complete,
            mode_transition_reason=self._last_transition_reason,
            completion_reason=self._completion_reason,
        )


__all__ = [
    "MissionMode",
    "MissionModeController",
    "MissionModeStepResult",
]
