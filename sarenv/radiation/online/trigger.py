"""Minimal evidence trigger for enabling radiation-priority decisions."""

from __future__ import annotations

import math
from typing import Protocol

from .measurement import RadiationMeasurement


RADIATION_CONFIRMATION_SAMPLES = 3


class RadiationAnomalyCriterion(Protocol):
    """Replaceable anomaly rule used by the consecutive-sample trigger."""

    def __call__(self, measurement: RadiationMeasurement) -> bool:
        """Return whether one measurement is anomalous."""
        ...


class AboveBackgroundCriterion:
    """The v1 rule: a measurement is anomalous only when above background."""

    def __init__(self, background_value: float, *, quantity: str, unit: str) -> None:
        if not math.isfinite(background_value) or background_value < 0.0:
            raise ValueError("background_value must be finite and non-negative.")
        if not quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not unit:
            raise ValueError("unit must be a non-empty string.")
        self.background_value = float(background_value)
        self.quantity = quantity
        self.unit = unit

    def __call__(self, measurement: RadiationMeasurement) -> bool:
        if not isinstance(measurement, RadiationMeasurement):
            raise TypeError("measurement must be a RadiationMeasurement.")
        if measurement.quantity != self.quantity or measurement.unit != self.unit:
            raise ValueError(
                "Measurement quantity/unit does not match the background contract."
            )
        return measurement.value > self.background_value


class ConsecutiveRadiationTrigger:
    """Latch after a configured run of anomalous measurements."""

    def __init__(
        self,
        criterion: RadiationAnomalyCriterion,
        *,
        confirmation_samples: int = RADIATION_CONFIRMATION_SAMPLES,
    ) -> None:
        if not callable(criterion):
            raise TypeError("criterion must be callable.")
        if isinstance(confirmation_samples, bool) or not isinstance(
            confirmation_samples, int
        ):
            raise TypeError("confirmation_samples must be an integer.")
        if confirmation_samples <= 0:
            raise ValueError("confirmation_samples must be positive.")
        self._criterion = criterion
        self.confirmation_samples = confirmation_samples
        self._consecutive_count = 0
        self._confirmed = False

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def observe(self, measurement: RadiationMeasurement) -> bool:
        """Consume one observation and return the latched confirmation state."""
        anomalous = self._criterion(measurement)
        if self._confirmed:
            return True
        if anomalous:
            self._consecutive_count += 1
            if self._consecutive_count >= self.confirmation_samples:
                self._confirmed = True
        else:
            self._consecutive_count = 0
        return self._confirmed

    def reset(self) -> None:
        """Explicitly reset state, for example at a new mission boundary."""
        self._consecutive_count = 0
        self._confirmed = False


__all__ = [
    "AboveBackgroundCriterion",
    "ConsecutiveRadiationTrigger",
    "RADIATION_CONFIRMATION_SAMPLES",
    "RadiationAnomalyCriterion",
]
