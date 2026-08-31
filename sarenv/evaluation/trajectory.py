"""Canonical continuous trajectory representation for post-run evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point


@dataclass(frozen=True)
class TrajectoryNode:
    """One ordered executed route node restored from a mission CSV."""

    step_index: int
    x_m: float
    y_m: float
    mode: str
    segment_distance_m: float
    cumulative_distance_m: float


@dataclass(frozen=True)
class SurvivorDiscovery:
    """Exact first contact between a route footprint and one survivor."""

    survivor_id: int
    found: bool
    first_discovery_distance_m: float | None
    exposure_end_distance_m: float


@dataclass(frozen=True)
class ExecutedTrajectory:
    """Ordered nodes interpreted as one continuous piecewise-linear route."""

    nodes: tuple[TrajectoryNode, ...]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("An executed trajectory requires at least one node.")
        previous_step = None
        previous_cumulative = None
        previous_xy = None
        for position, node in enumerate(self.nodes):
            numeric = (
                node.x_m,
                node.y_m,
                node.segment_distance_m,
                node.cumulative_distance_m,
            )
            if not np.isfinite(numeric).all():
                raise ValueError("Trajectory coordinates and distances must be finite.")
            if node.segment_distance_m < 0.0 or node.cumulative_distance_m < 0.0:
                raise ValueError("Trajectory distances must be non-negative.")
            if not node.mode:
                raise ValueError("Every trajectory node requires a mode.")
            if previous_step is not None and node.step_index <= previous_step:
                raise ValueError("Trajectory step indices must be strictly increasing.")
            if previous_cumulative is not None:
                if node.cumulative_distance_m + 1e-7 < previous_cumulative:
                    raise ValueError("Cumulative route distance must be monotonic.")
                expected = math.dist(previous_xy, (node.x_m, node.y_m))
                if not math.isclose(
                    node.segment_distance_m,
                    expected,
                    rel_tol=1e-8,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        f"Segment distance at node {position} does not match coordinates."
                    )
                expected_cumulative = previous_cumulative + expected
                if not math.isclose(
                    node.cumulative_distance_m,
                    expected_cumulative,
                    rel_tol=1e-8,
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        f"Cumulative distance at node {position} is inconsistent."
                    )
            previous_step = node.step_index
            previous_cumulative = node.cumulative_distance_m
            previous_xy = (node.x_m, node.y_m)

    @cached_property
    def coordinates(self) -> tuple[tuple[float, float], ...]:
        return tuple((node.x_m, node.y_m) for node in self.nodes)

    @cached_property
    def line(self) -> LineString:
        coordinates = self.coordinates
        if len(coordinates) == 1:
            coordinates = (coordinates[0], coordinates[0])
        return LineString(coordinates)

    @property
    def total_distance_m(self) -> float:
        return float(self.nodes[-1].cumulative_distance_m)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def segment_count(self) -> int:
        return max(0, len(self.nodes) - 1)

    def interpolate(self, distance_m: float) -> Point:
        if not math.isfinite(distance_m):
            raise ValueError("distance_m must be finite.")
        clipped = min(max(float(distance_m), 0.0), self.total_distance_m)
        if self.total_distance_m == 0.0:
            first = self.nodes[0]
            return Point(first.x_m, first.y_m)
        return self.line.interpolate(clipped)

    def sample_coordinates(self, distances_m: np.ndarray) -> np.ndarray:
        """Interpolate many route distances without changing path semantics."""
        distances = np.asarray(distances_m, dtype=float)
        if distances.ndim != 1:
            raise ValueError("distances_m must be a one-dimensional array.")
        if not np.isfinite(distances).all():
            raise ValueError("distances_m must contain only finite values.")
        if distances.size == 0:
            return np.empty((0, 2), dtype=float)

        coordinates = np.asarray(self.coordinates, dtype=float)
        if self.total_distance_m == 0.0 or coordinates.shape[0] == 1:
            return np.repeat(coordinates[:1], distances.size, axis=0)

        segment_vectors = np.diff(coordinates, axis=0)
        segment_lengths = np.hypot(
            segment_vectors[:, 0],
            segment_vectors[:, 1],
        )
        cumulative_lengths = np.concatenate(
            (np.asarray([0.0]), np.cumsum(segment_lengths))
        )
        clipped = np.clip(distances, 0.0, self.total_distance_m)
        # ``LineString.interpolate`` clamps to the geometry length.  Preserve that
        # behaviour when the saved cumulative distance differs by round-off only.
        clipped = np.minimum(clipped, cumulative_lengths[-1])
        segment_indices = np.searchsorted(
            cumulative_lengths,
            clipped,
            side="right",
        ) - 1
        segment_indices = np.clip(segment_indices, 0, segment_lengths.size - 1)

        selected_lengths = segment_lengths[segment_indices]
        offsets = clipped - cumulative_lengths[segment_indices]
        fractions = np.divide(
            offsets,
            selected_lengths,
            out=np.zeros_like(offsets),
            where=selected_lengths > 0.0,
        )
        return coordinates[segment_indices] + (
            fractions[:, np.newaxis] * segment_vectors[segment_indices]
        )


def load_executed_trajectory(path: str | Path) -> ExecutedTrajectory:
    """Load and validate the canonical fields in ``mission_steps.csv``."""
    source_path = Path(path).resolve()
    required = {
        "step_index",
        "mode_before",
        "x_m",
        "y_m",
        "step_distance_m",
        "total_distance_m",
    }
    with source_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"Trajectory CSV is missing required fields: {missing}")
        nodes = tuple(
            TrajectoryNode(
                step_index=int(row["step_index"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                mode=str(row["mode_before"]),
                segment_distance_m=float(row["step_distance_m"]),
                cumulative_distance_m=float(row["total_distance_m"]),
            )
            for row in reader
        )
    trajectory = ExecutedTrajectory(nodes=nodes, source_path=source_path)
    if not math.isclose(
        trajectory.line.length,
        trajectory.total_distance_m,
        rel_tol=1e-9,
        abs_tol=1e-5,
    ):
        raise ValueError("Trajectory polyline length does not match saved distance.")
    return trajectory


def first_discovery_distance_m(
    trajectory: ExecutedTrajectory,
    survivor: Point,
    detection_radius_m: float,
) -> float | None:
    """Return the exact first route distance entering a circular footprint."""
    if not isinstance(survivor, Point) or survivor.is_empty:
        raise TypeError("survivor must be a non-empty Point.")
    if not math.isfinite(detection_radius_m) or detection_radius_m < 0.0:
        raise ValueError("detection_radius_m must be finite and non-negative.")

    coordinates = trajectory.coordinates
    radius_squared = float(detection_radius_m) ** 2
    start_dx = coordinates[0][0] - survivor.x
    start_dy = coordinates[0][1] - survivor.y
    if start_dx * start_dx + start_dy * start_dy <= radius_squared:
        return 0.0

    cumulative = 0.0
    for start, end in zip(coordinates, coordinates[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length == 0.0:
            continue
        offset_x = start[0] - survivor.x
        offset_y = start[1] - survivor.y
        quadratic_a = dx * dx + dy * dy
        quadratic_b = 2.0 * (offset_x * dx + offset_y * dy)
        quadratic_c = offset_x * offset_x + offset_y * offset_y - radius_squared
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        if discriminant >= 0.0:
            root = math.sqrt(max(0.0, discriminant))
            enter_fraction = (-quadratic_b - root) / (2.0 * quadratic_a)
            exit_fraction = (-quadratic_b + root) / (2.0 * quadratic_a)
            if exit_fraction >= 0.0 and enter_fraction <= 1.0:
                clipped_fraction = min(max(enter_fraction, 0.0), 1.0)
                return float(cumulative + clipped_fraction * segment_length)
        cumulative += segment_length
    return None


def discover_survivors(
    trajectory: ExecutedTrajectory,
    survivors: tuple[Point, ...] | list[Point],
    detection_radius_m: float,
) -> tuple[SurvivorDiscovery, ...]:
    """Evaluate exact first discovery distance for every survivor."""
    results = []
    for survivor_id, survivor in enumerate(survivors):
        discovery = first_discovery_distance_m(
            trajectory,
            survivor,
            detection_radius_m,
        )
        results.append(
            SurvivorDiscovery(
                survivor_id=survivor_id,
                found=discovery is not None,
                first_discovery_distance_m=discovery,
                exposure_end_distance_m=(
                    trajectory.total_distance_m if discovery is None else discovery
                ),
            )
        )
    return tuple(results)


def calculate_turn_diagnostics(
    trajectory: ExecutedTrajectory,
) -> dict[str, float | int]:
    """Calculate geometric heading-change diagnostics without time penalties."""
    headings = []
    for start, end in zip(
        trajectory.coordinates,
        trajectory.coordinates[1:],
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dx != 0.0 or dy != 0.0:
            headings.append(math.degrees(math.atan2(dy, dx)))
    changes = []
    for previous, current in zip(headings, headings[1:]):
        signed = (current - previous + 180.0) % 360.0 - 180.0
        changes.append(abs(signed))
    changes_array = np.asarray(changes, dtype=float)
    return {
        "number_of_segments": len(headings),
        "number_of_heading_changes": int(np.count_nonzero(changes_array > 1e-12)),
        "total_absolute_heading_change_deg": float(changes_array.sum()),
        "mean_heading_change_deg": (
            float(changes_array.mean()) if changes_array.size else 0.0
        ),
        "number_of_turns_above_45_deg": int(
            np.count_nonzero(changes_array > 45.0)
        ),
        "number_of_turns_above_90_deg": int(
            np.count_nonzero(changes_array > 90.0)
        ),
    }


__all__ = [
    "ExecutedTrajectory",
    "SurvivorDiscovery",
    "TrajectoryNode",
    "calculate_turn_diagnostics",
    "discover_survivors",
    "first_discovery_distance_m",
    "load_executed_trajectory",
]
