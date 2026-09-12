"""SAR-gated radiation-priority decisions using online excess estimates."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from numbers import Integral
import time
from typing import Literal, Sequence

import numpy as np

from sarenv.analytics.paths import (
    greedy_grid_cell_bounds,
    greedy_grid_position_to_world,
    greedy_searchable_cells,
    greedy_visible_cells,
    score_greedy_neighbours,
    select_original_greedy_candidate,
)
from sarenv.radiation.online.interfaces import OnlineRadiationEstimator
from sarenv.radiation.online.measurement import RadiationRegionEstimate


SAR_REFERENCE_RADIUS_FRACTION = 0.5


@dataclass(frozen=True)
class SarReferenceScore:
    """Resolved native Greedy score used as one SAR relevance unit."""

    score: float
    radius_fraction: float
    radius_m: float
    ring_cell_count: int
    method: str = "minimum_native_visible_probability_on_reference_ring"


def derive_sar_reference_score(
    *,
    probability_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    search_center: tuple[float, float],
    max_radius: float,
    detection_radius_m: float,
    reference_radius_fraction: float = SAR_REFERENCE_RADIUS_FRACTION,
) -> SarReferenceScore:
    """Resolve a scenario-specific reference from native visible-cell scores."""
    probability_map = np.asarray(probability_map)
    if probability_map.ndim != 2 or probability_map.size == 0:
        raise ValueError("probability_map must be a non-empty 2D array.")
    if not np.isfinite(probability_map).all() or np.any(probability_map < 0.0):
        raise ValueError("probability_map must contain finite non-negative values.")
    if not math.isfinite(reference_radius_fraction) or not (
        0.0 < reference_radius_fraction < 1.0
    ):
        raise ValueError("reference_radius_fraction must lie strictly between 0 and 1.")
    if not math.isfinite(max_radius) or max_radius <= 0.0:
        raise ValueError("max_radius must be finite and positive.")
    if not math.isfinite(detection_radius_m) or detection_radius_m < 0.0:
        raise ValueError("detection_radius_m must be finite and non-negative.")

    height, width = probability_map.shape
    minx, miny, maxx, maxy = (float(value) for value in bounds)
    dx = (maxx - minx) / width
    dy = (maxy - miny) / height
    ring_tolerance_m = 0.5 * math.hypot(dx, dy)
    reference_radius_m = float(max_radius) * float(reference_radius_fraction)
    searchable = greedy_searchable_cells(
        map_shape=probability_map.shape,
        bounds=bounds,
        search_center=search_center,
        max_radius=max_radius,
    )

    def radial_error(cell: tuple[int, int]) -> float:
        x_m, y_m = greedy_grid_position_to_world(
            *cell,
            map_shape=probability_map.shape,
            bounds=bounds,
        )
        return abs(math.dist((x_m, y_m), search_center) - reference_radius_m)

    ring_cells = [
        cell for cell in searchable if radial_error(cell) <= ring_tolerance_m
    ]
    if not ring_cells and searchable:
        smallest_error = min(radial_error(cell) for cell in searchable)
        ring_cells = [
            cell
            for cell in searchable
            if math.isclose(
                radial_error(cell),
                smallest_error,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]
    if not ring_cells:
        raise ValueError("No searchable SAR cells exist for reference derivation.")

    native_scores = []
    for row, col in ring_cells:
        visible = greedy_visible_cells(
            row,
            col,
            map_shape=probability_map.shape,
            bounds=bounds,
            detection_radius_m=detection_radius_m,
        )
        native_scores.append(sum(float(probability_map[r, c]) for r, c in visible))
    # The gate means "at least as worth searching as the weakest native SAR
    # opportunity on the configured reference ring".  Using the minimum keeps
    # this a conservative relevance gate derived entirely from SAR truth; it
    # neither reads the radiation scenario nor introduces a probability magic
    # number.  The reference score remains the priority scale in the radiation
    # equivalence calculation.
    return SarReferenceScore(
        score=float(np.min(native_scores)),
        radius_fraction=float(reference_radius_fraction),
        radius_m=reference_radius_m,
        ring_cell_count=len(ring_cells),
    )


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
    x_m, y_m = greedy_grid_position_to_world(
        row,
        col,
        map_shape=map_shape,
        bounds=bounds,
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
    *,
    bounds: tuple[float, float, float, float],
    map_shape: tuple[int, int],
) -> RadiationRegionEstimate:
    """Return the MAX estimate over the candidate SAR cell's full bounds."""
    cell_bounds = greedy_grid_cell_bounds(
        candidate.row,
        candidate.col,
        map_shape=map_shape,
        bounds=bounds,
    )
    return estimator.max_estimate_in_bounds(cell_bounds)


@dataclass(frozen=True)
class RadiationCandidateAssessment:
    """Resolved SAR and radiation values for one local candidate."""

    candidate: GreedyCandidate
    sar_relevant: bool
    radiation_estimate: RadiationRegionEstimate
    radiation_equivalent_priority: float
    chosen: bool = False
    selection_reason: str = "not_selected"


@dataclass(frozen=True)
class RadiationPriorityDecision:
    """One radiation-aware or Original-Greedy local selection."""

    planner_mode: Literal["original_greedy", "radiation_priority"]
    reason: str
    selected_candidate: GreedyCandidate | None = None
    radiation_estimate: RadiationRegionEstimate | None = None
    assessments: tuple[RadiationCandidateAssessment, ...] = ()

    @property
    def fallback_to_original(self) -> bool:
        return self.planner_mode == "original_greedy"


class RadiationPriorityGreedyPolicy:
    """Apply derived SAR relevance and SAR-equivalent radiation ranking."""

    def __init__(
        self,
        *,
        sar_reference_score: float,
        hazard_reference_excess: float = 1.0,
    ) -> None:
        if not math.isfinite(sar_reference_score) or sar_reference_score < 0.0:
            raise ValueError("sar_reference_score must be finite and non-negative.")
        if not math.isfinite(hazard_reference_excess) or hazard_reference_excess <= 0.0:
            raise ValueError("hazard_reference_excess must be finite and positive.")
        self.sar_reference_score = float(sar_reference_score)
        self.hazard_reference_excess = float(hazard_reference_excess)

    def select(
        self,
        candidates: Sequence[GreedyCandidate],
        estimator: OnlineRadiationEstimator,
        *,
        radiation_priority_enabled: bool,
        bounds: tuple[float, float, float, float],
        map_shape: tuple[int, int],
    ) -> RadiationPriorityDecision:
        """Rank SAR-relevant candidates by conservative radiation MAX."""
        assessments = []
        for candidate in candidates:
            estimate = estimate_candidate_radiation(
                candidate,
                estimator,
                bounds=bounds,
                map_shape=map_shape,
            )
            sar_relevant = candidate.sar_score >= self.sar_reference_score
            equivalent_priority = (
                self.sar_reference_score
                * estimate.value
                / self.hazard_reference_excess
            )
            assessments.append(
                RadiationCandidateAssessment(
                    candidate=candidate,
                    sar_relevant=sar_relevant,
                    radiation_estimate=estimate,
                    radiation_equivalent_priority=float(equivalent_priority),
                )
            )

        if not radiation_priority_enabled:
            return RadiationPriorityDecision(
                planner_mode="original_greedy",
                reason="radiation_not_active",
                assessments=tuple(assessments),
            )

        eligible = [assessment for assessment in assessments if assessment.sar_relevant]
        if not eligible:
            return RadiationPriorityDecision(
                planner_mode="original_greedy",
                reason="no_candidate_meets_sar_reference",
                assessments=tuple(assessments),
            )

        metadata = {
            (item.radiation_estimate.quantity, item.radiation_estimate.unit)
            for item in eligible
        }
        if len(metadata) != 1:
            raise ValueError("Candidate estimates do not share one quantity/unit contract.")
        selected = max(
            eligible,
            key=lambda item: (
                item.radiation_equivalent_priority,
                item.candidate.sar_score,
            ),
        )
        resolved = tuple(
            replace(
                item,
                chosen=item is selected,
                selection_reason=(
                    "highest_radiation_equivalent_priority"
                    if item is selected
                    else "lower_radiation_equivalent_priority"
                ),
            )
            for item in assessments
        )
        return RadiationPriorityDecision(
            planner_mode="radiation_priority",
            reason="radiation_priority",
            selected_candidate=selected.candidate,
            radiation_estimate=selected.radiation_estimate,
            assessments=resolved,
        )


@dataclass(frozen=True)
class RadiationGreedyStepPlan:
    """One local planning result; it does not precompute later steps."""

    candidates: tuple[GreedyCandidate, ...]
    decision: RadiationPriorityDecision
    planning_time_s: float

    @property
    def selected_candidate(self) -> GreedyCandidate | None:
        return self.decision.selected_candidate

    @property
    def normal_replan_required(self) -> bool:
        return self.selected_candidate is None


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
        hazard_reference_excess: float = 1.0,
        reference_radius_fraction: float = SAR_REFERENCE_RADIUS_FRACTION,
        sar_reference: SarReferenceScore | None = None,
        policy: RadiationPriorityGreedyPolicy | None = None,
        rng: np.random.Generator | None = None,
        fallback_within_radiation: bool = True,
    ) -> None:
        probability_map = np.asarray(probability_map)
        if probability_map.ndim != 2 or probability_map.size == 0:
            raise ValueError("probability_map must be a non-empty 2D array.")
        if not np.isfinite(probability_map).all() or np.any(probability_map < 0.0):
            raise ValueError("probability_map must contain finite non-negative values.")
        if not callable(getattr(estimator, "max_estimate_in_bounds", None)):
            raise TypeError("estimator must provide max_estimate_in_bounds().")
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator.")

        self.probability_map = probability_map
        self.bounds = tuple(float(value) for value in bounds)
        self.search_center = tuple(float(value) for value in search_center)
        self.max_radius = float(max_radius)
        self.detection_radius_m = float(detection_radius_m)
        self.estimator = estimator
        self.rng = rng
        self.fallback_within_radiation = bool(fallback_within_radiation)
        self.sar_reference = sar_reference or derive_sar_reference_score(
            probability_map=probability_map,
            bounds=self.bounds,
            search_center=self.search_center,
            max_radius=self.max_radius,
            detection_radius_m=self.detection_radius_m,
            reference_radius_fraction=reference_radius_fraction,
        )
        self.policy = policy or RadiationPriorityGreedyPolicy(
            sar_reference_score=self.sar_reference.score,
            hazard_reference_excess=hazard_reference_excess,
        )

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
            bounds=self.bounds,
            map_shape=self.probability_map.shape,
        )
        if (
            radiation_priority_enabled
            and decision.selected_candidate is None
            and candidates
            and self.fallback_within_radiation
        ):
            selected_position = select_original_greedy_candidate(scored, self.rng)
            selected_candidate = next(
                candidate
                for candidate in candidates
                if (candidate.row, candidate.col) == selected_position
            )
            selected_assessment = next(
                item
                for item in decision.assessments
                if item.candidate == selected_candidate
            )
            assessments = tuple(
                replace(
                    item,
                    chosen=item.candidate == selected_candidate,
                    selection_reason=(
                        "original_greedy_fallback"
                        if item.candidate == selected_candidate
                        else "not_selected_by_original_greedy_fallback"
                    ),
                )
                for item in decision.assessments
            )
            decision = RadiationPriorityDecision(
                planner_mode="original_greedy",
                reason=f"{decision.reason}_original_greedy_step",
                selected_candidate=selected_candidate,
                radiation_estimate=selected_assessment.radiation_estimate,
                assessments=assessments,
            )
        planning_time_s = time.perf_counter() - planning_started
        return RadiationGreedyStepPlan(
            candidates=candidates,
            decision=decision,
            planning_time_s=planning_time_s,
        )


__all__ = [
    "GreedyCandidate",
    "RadiationCandidateAssessment",
    "RadiationGreedyStepPlan",
    "RadiationGreedyStepPlanner",
    "RadiationPriorityDecision",
    "RadiationPriorityGreedyPolicy",
    "SAR_REFERENCE_RADIUS_FRACTION",
    "SarReferenceScore",
    "derive_sar_reference_score",
    "estimate_candidate_radiation",
    "greedy_candidate_from_grid",
]
