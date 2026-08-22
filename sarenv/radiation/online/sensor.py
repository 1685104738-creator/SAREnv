"""Minimal scalar radiation sensor for benchmark observations."""

from __future__ import annotations

from .interfaces import RadiationTruthQuery
from .measurement import RadiationMeasurement


class NoiseFreeRadiationSensor:
    """Query scalar truth at the UAV position without adding detector noise."""

    def __init__(
        self,
        truth_query: RadiationTruthQuery,
        *,
        quantity: str,
        unit: str,
    ) -> None:
        if not callable(truth_query):
            raise TypeError("truth_query must be callable.")
        if not quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not unit:
            raise ValueError("unit must be a non-empty string.")
        self._truth_query = truth_query
        self.quantity = quantity
        self.unit = unit

    def measure(
        self,
        x_m: float,
        y_m: float,
        altitude_m: float,
        simulated_time_s: float,
    ) -> RadiationMeasurement:
        """Return one scalar truth observation with its sampling coordinates."""
        value = float(self._truth_query(x_m, y_m, altitude_m))
        return RadiationMeasurement(
            x_m=float(x_m),
            y_m=float(y_m),
            altitude_m=float(altitude_m),
            simulated_time_s=float(simulated_time_s),
            value=value,
            quantity=self.quantity,
            unit=self.unit,
        )


__all__ = ["NoiseFreeRadiationSensor"]
