"""Small planner-facing contracts for radiation-priority Greedy search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import time
from typing import Literal, Sequence

import numpy as np

from sarenv.analytics.paths import (
    greedy_grid_position_to_world,
    score_greedy_neighbours,
)
from sarenv.radiation.online.interfaces import OnlineRadiationEstimator
from sarenv.radiation.online.measurement import RadiationEstimate


SAR_PROBABILITY_GATE = 0.001


@dataclass(frozen=True)
class GreedyCandidate:
    """One existing SAR Greedy neighbour with its cell-centre position."""

    row: int
    col: int
    x_m: float
    y_m: float
    sar_score: float

    def __post_init__(self) -> None:
        if isinstance(self.row, bool) or not isinstance(self.row, Integral):
            raise TypeError("row must be an integer.")
        if isinstance(self.col, bool) or not isinstance(self.col, Integral):
            raise TypeError("col must be an integer.")
        if self.row < 0 or self.col < 0:
            raise ValueError("row and col must be non-negative.")
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "col", int(self.col))
        if not all(math.isfinite(value) for value in (self.x_m, self.y_m)):
            raise ValueError("Candidate coordinates must be finite.")
        if not math.isfinite(self.sar_score) or self.sar_score < 0.0:
            raise ValueError("sar_score must be finite and non-negative.")


def greedy_candidate_from_grid(
    row: int,
    col: int,
    sar_score: float,
    *,
    bounds: tuple[float, float, float, float],
    map_shape: tuple[int, int],
) -> GreedyCandidate:
    """Attach the exact cell centre used by ``generate_greedy_path``."""
    height, width = map_shape
    if height <= 0 or width <= 0:
        raise ValueError("map_shape dimensions must be positive.")
    if not 0 <= row < height or not 0 <= col < width:
        raise IndexError("Candidate row/col lies outside map_shape.")

    minx, miny, maxx, maxy = (float(value) for value in bounds)
    if not all(math.isfinite(value) for value in (minx, miny, maxx, maxy)):
        raise ValueError("bounds must contain four finite values.")
    if maxx <= minx or maxy <= miny:
        raise ValueError("bounds must have positive width and height.")

    x_m, y_m = greedy_grid_position_to_world(
        row,
        col,
        map_shape=map_shape,
        bounds=(minx, miny, maxx, maxy),
    )
    return GreedyCandidate(
        row=row,
        col=col,
        x_m=x_m,
        y_m=y_m,
        sar_score=float(sar_score),
    )


def estimate_candidate_radiation(
    candidate: GreedyCandidate,
    estimator: OnlineRadiationEstimator,
) -> RadiationEstimate:
    """Query online radiation at an existing SAR candidate's centre."""
    return estimator.estimate_at(candidate.x_m, candidate.y_m)


@dataclass(frozen=True)
class RadiationPriorityDecision:
    """Result of a promotion attempt; fallback leaves selection to Original Greedy."""

    planner_mode: Literal["original_greedy", "radiation_priority"]
    reason: str
    selected_candidate: GreedyCandidate | None = None
    radiation_estimate: RadiationEstimate | None = None

    @property
    def fallback_to_original(self) -> bool:
        return self.planner_mode == "original_greedy"


class RadiationPriorityGreedyPolicy:
    """Apply the v1 SAR-gated lexicographic promotion rule."""

    def __init__(self, *, sar_probability_gate: float = SAR_PROBABILITY_GATE):
        if not math.isfinite(sar_probability_gate) or sar_probability_gate < 0.0:
            raise ValueError("sar_probability_gate must be finite and non-negative.")
        self.sar_probability_gate = float(sar_probability_gate)

    def select(
        self,
        candidates: Sequence[GreedyCandidate],
        estimator: OnlineRadiationEstimator,
        *,
        radiation_priority_enabled: bool,
    ) -> RadiationPriorityDecision:
        """Select a promoted existing candidate or request Original Greedy fallback."""
        if not radiation_priority_enabled:
            return self._fallback("radiation_not_confirmed")

        eligible = [
            candidate
            for candidate in candidates
            if candidate.sar_score >= self.sar_probability_gate
        ]
        if not eligible:
            return self._fallback("no_candidate_meets_sar_gate")

        estimates = [
            (candidate, estimate_candidate_radiation(candidate, estimator))
            for candidate in eligible
        ]
        known = [pair for pair in estimates if pair[1].observed]
        if not known:
            return self._fallback("no_usable_radiation_estimate")
        if len(known) != len(estimates):
            return self._fallback("partial_radiation_estimates")

        metadata = {(estimate.quantity, estimate.unit) for _, estimate in known}
        if len(metadata) != 1:
            raise ValueError("Candidate estimates do not share one quantity/unit contract.")

        selected_candidate, selected_estimate = max(
            known,
            key=lambda pair: (
                float(pair[1].value),
                pair[0].sar_score,
            ),
        )
        return RadiationPriorityDecision(
            planner_mode="radiation_priority",
            reason="radiation_priority",
            selected_candidate=selected_candidate,
            radiation_estimate=selected_estimate,
        )

    @staticmethod
    def _fallback(reason: str) -> RadiationPriorityDecision:
        return RadiationPriorityDecision(
            planner_mode="original_greedy",
            reason=reason,
        )


@dataclass(frozen=True)
class RadiationGreedyStepPlan:
    """One local planning result; it does not execute or precompute later steps."""

    candidates: tuple[GreedyCandidate, ...]
    decision: RadiationPriorityDecision
    planning_time_s: float

    @property
    def selected_candidate(self) -> GreedyCandidate | None:
        return self.decision.selected_candidate

    @property
    def normal_replan_required(self) -> bool:
        return self.decision.fallback_to_original


class RadiationGreedyStepPlanner:
    """Score and radiation-rank only the current cell's eight neighbours."""

    def __init__(
        self,
        *,
        probability_map: np.ndarray,
        bounds: tuple[float, float, float, float],
        search_center: tuple[float, float],
        max_radius: float,
        detection_radius_m: float,
        estimator: OnlineRadiationEstimator,
        policy: RadiationPriorityGreedyPolicy | None = None,
    ) -> None:
        probability_map = np.asarray(probability_map)
        if probability_map.ndim != 2 or probability_map.size == 0:
            raise ValueError("probability_map must be a non-empty 2D array.")
        if not np.isfinite(probability_map).all() or np.any(probability_map < 0.0):
            raise ValueError("probability_map must contain finite non-negative values.")
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("bounds must contain four finite values.")
        if len(search_center) != 2 or not all(
            math.isfinite(value) for value in search_center
        ):
            raise ValueError("search_center must contain two finite values.")
        if not math.isfinite(max_radius) or max_radius <= 0.0:
            raise ValueError("max_radius must be finite and positive.")
        if not math.isfinite(detection_radius_m) or detection_radius_m < 0.0:
            raise ValueError("detection_radius_m must be finite and non-negative.")
        if not callable(getattr(estimator, "estimate_at", None)):
            raise TypeError("estimator must provide estimate_at().")

        self.probability_map = probability_map
        self.bounds = tuple(float(value) for value in bounds)
        self.search_center = tuple(float(value) for value in search_center)
        self.max_radius = float(max_radius)
        self.detection_radius_m = float(detection_radius_m)
        self.estimator = estimator
        self.policy = policy or RadiationPriorityGreedyPolicy()

    def plan_next(
        self,
        current_position: tuple[int, int],
        observed_cells: set[tuple[int, int]] | frozenset[tuple[int, int]],
        *,
        radiation_priority_enabled: bool,
    ) -> RadiationGreedyStepPlan:
        """Return at most one next SAR cell from the local neighbour set."""
        planning_started = time.perf_counter()
        scored = score_greedy_neighbours(
            current_position,
            set(observed_cells),
            probability_map=self.probability_map,
            bounds=self.bounds,
            search_center=self.search_center,
            max_radius=self.max_radius,
            detection_radius_m=self.detection_radius_m,
        )
        candidates = tuple(
            greedy_candidate_from_grid(
                row,
                col,
                score,
                bounds=self.bounds,
                map_shape=self.probability_map.shape,
            )
            for (row, col), score in scored
        )
        decision = self.policy.select(
            candidates,
            self.estimator,
            radiation_priority_enabled=radiation_priority_enabled,
        )
        planning_time_s = time.perf_counter() - planning_started
        return RadiationGreedyStepPlan(
            candidates=candidates,
            decision=decision,
            planning_time_s=planning_time_s,
        )


__all__ = [
    "GreedyCandidate",
    "RadiationPriorityDecision",
    "RadiationPriorityGreedyPolicy",
    "RadiationGreedyStepPlan",
    "RadiationGreedyStepPlanner",
    "SAR_PROBABILITY_GATE",
    "estimate_candidate_radiation",
    "greedy_candidate_from_grid",
]
