"""Small dependency boundaries for the online radiation pipeline."""

from __future__ import annotations

from typing import Protocol

from .measurement import RadiationEstimate, RadiationMeasurement


class RadiationTruthQuery(Protocol):
    """Truth access available to a sensor, never to a planner policy."""

    def __call__(self, x_m: float, y_m: float, altitude_m: float) -> float:
        """Return one scalar truth value at a three-dimensional position."""
        ...


class RadiationMeasurementSource(Protocol):
    """A sensor-facing source of scalar radiation measurements."""

    def measure(
        self,
        x_m: float,
        y_m: float,
        altitude_m: float,
        simulated_time_s: float,
    ) -> RadiationMeasurement:
        """Produce one measurement without exposing the underlying truth object."""
        ...


class OnlineRadiationEstimator(Protocol):
    """The only radiation-map interface intended for planner consumption."""

    def update(self, measurement: RadiationMeasurement) -> int:
        """Assimilate one measurement and return the number of updated cells."""
        ...

    def estimate_at(self, x_m: float, y_m: float) -> RadiationEstimate:
        """Return a known or explicitly unknown estimate at a position."""
        ...


__all__ = [
    "OnlineRadiationEstimator",
    "RadiationMeasurementSource",
    "RadiationTruthQuery",
]
