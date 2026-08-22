"""Simulation-time primitives independent of wall-clock execution time."""

from __future__ import annotations

import math


class MissionClock:
    """Advance deterministic mission time from travelled distance and speed."""

    def __init__(self, *, initial_time_s: float = 0.0) -> None:
        if not math.isfinite(initial_time_s) or initial_time_s < 0.0:
            raise ValueError("initial_time_s must be finite and non-negative.")
        self._time_s = float(initial_time_s)

    @property
    def time_s(self) -> float:
        return self._time_s

    def advance_travel(self, distance_m: float, *, uav_speed_m_s: float) -> float:
        """Advance by ``distance_m / uav_speed_m_s`` and return current time."""
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError("distance_m must be finite and non-negative.")
        if not math.isfinite(uav_speed_m_s) or uav_speed_m_s <= 0.0:
            raise ValueError("uav_speed_m_s must be finite and positive.")
        self._time_s += float(distance_m) / float(uav_speed_m_s)
        return self._time_s


__all__ = ["MissionClock"]
