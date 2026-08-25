"""Minimal scalar radiation sensor for benchmark observations."""

from __future__ import annotations

import math

from .interfaces import RadiationTruthQuery
from .measurement import RadiationMeasurement


class NoiseFreeRadiationSensor:
    """Query one scalar truth value, optionally on a corrected reference plane."""

    def __init__(
        self,
        truth_query: RadiationTruthQuery,
        *,
        quantity: str,
        unit: str,
        value_reference_height_m: float | None = None,
        background_value_to_subtract: float = 0.0,
    ) -> None:
        if not callable(truth_query):
            raise TypeError("truth_query must be callable.")
        if not quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not unit:
            raise ValueError("unit must be a non-empty string.")
        if value_reference_height_m is not None and (
            not math.isfinite(value_reference_height_m)
            or value_reference_height_m < 0.0
        ):
            raise ValueError(
                "value_reference_height_m must be finite and non-negative."
            )
        if (
            not math.isfinite(background_value_to_subtract)
            or background_value_to_subtract < 0.0
        ):
            raise ValueError(
                "background_value_to_subtract must be finite and non-negative."
            )
        self._truth_query = truth_query
        self.quantity = quantity
        self.unit = unit
        self.value_reference_height_m = value_reference_height_m
        self.background_value_to_subtract = float(background_value_to_subtract)

    def measure(
        self,
        x_m: float,
        y_m: float,
        platform_altitude_m: float | None = None,
        simulated_time_s: float = 0.0,
        *,
        altitude_m: float | None = None,
    ) -> RadiationMeasurement:
        """Return a scalar observation, optionally referenced to another height."""
        if platform_altitude_m is None:
            if altitude_m is None:
                raise TypeError("platform_altitude_m must be provided.")
            platform_altitude_m = altitude_m
        elif altitude_m is not None and not math.isclose(
            platform_altitude_m,
            altitude_m,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("platform_altitude_m and altitude_m disagree.")
        reference_height_m = (
            float(platform_altitude_m)
            if self.value_reference_height_m is None
            else self.value_reference_height_m
        )
        truth_value = float(self._truth_query(x_m, y_m, reference_height_m))
        value = max(truth_value - self.background_value_to_subtract, 0.0)
        return RadiationMeasurement(
            x_m=float(x_m),
            y_m=float(y_m),
            platform_altitude_m=float(platform_altitude_m),
            value_reference_height_m=reference_height_m,
            simulated_time_s=float(simulated_time_s),
            value=value,
            quantity=self.quantity,
            unit=self.unit,
        )


__all__ = ["NoiseFreeRadiationSensor"]
