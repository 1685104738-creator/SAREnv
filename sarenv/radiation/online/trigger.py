"""Minimal evidence trigger for enabling radiation-priority decisions."""

from __future__ import annotations

import math
from typing import Protocol

from .measurement import RadiationMeasurement


RADIATION_CONFIRMATION_SAMPLES = 3
NOISE_FREE_CONFIRMATION_SAMPLES = 1
NOMINAL_HYSTERESIS_MARGIN = 0.10


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
        self._last_transition: str | None = None

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    @property
    def last_transition(self) -> str | None:
        return self._last_transition

    def observe(self, measurement: RadiationMeasurement) -> bool:
        """Consume one observation and return the latched confirmation state."""
        self._last_transition = None
        anomalous = self._criterion(measurement)
        if self._confirmed:
            return True
        if anomalous:
            self._consecutive_count += 1
            if self._consecutive_count >= self.confirmation_samples:
                self._confirmed = True
                self._last_transition = "enter"
        else:
            self._consecutive_count = 0
        return self._confirmed

    def reset(self) -> None:
        """Explicitly reset state, for example at a new mission boundary."""
        self._consecutive_count = 0
        self._confirmed = False
        self._last_transition = None


class HysteresisRadiationTrigger:
    """Stateful hazard trigger with one derived 10% enter/exit band."""

    def __init__(
        self,
        *,
        enter_threshold: float,
        quantity: str,
        unit: str,
        hysteresis_margin: float = NOMINAL_HYSTERESIS_MARGIN,
        confirmation_samples: int = NOISE_FREE_CONFIRMATION_SAMPLES,
    ) -> None:
        if not math.isfinite(enter_threshold) or enter_threshold <= 0.0:
            raise ValueError("enter_threshold must be finite and positive.")
        if not math.isfinite(hysteresis_margin) or not (
            0.0 < hysteresis_margin < 1.0
        ):
            raise ValueError("hysteresis_margin must lie strictly between 0 and 1.")
        if isinstance(confirmation_samples, bool) or not isinstance(
            confirmation_samples,
            int,
        ):
            raise TypeError("confirmation_samples must be an integer.")
        if confirmation_samples <= 0:
            raise ValueError("confirmation_samples must be positive.")
        if not quantity or not unit:
            raise ValueError("quantity and unit must be non-empty.")
        self.enter_threshold = float(enter_threshold)
        self.hysteresis_margin = float(hysteresis_margin)
        self.exit_threshold = (1.0 - self.hysteresis_margin) * self.enter_threshold
        self.confirmation_samples = confirmation_samples
        self.quantity = quantity
        self.unit = unit
        self._consecutive_count = 0
        self._confirmed = False
        self._last_transition: str | None = None

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count

    @property
    def confirmed(self) -> bool:
        """Return whether the hazard state is currently active."""
        return self._confirmed

    @property
    def last_transition(self) -> str | None:
        return self._last_transition

    def observe(self, measurement: RadiationMeasurement) -> bool:
        """Apply enter/exit thresholds and retain state inside the hysteresis band."""
        if not isinstance(measurement, RadiationMeasurement):
            raise TypeError("measurement must be a RadiationMeasurement.")
        if measurement.quantity != self.quantity or measurement.unit != self.unit:
            raise ValueError(
                "Measurement quantity/unit does not match the hazard contract."
            )
        self._last_transition = None
        if self._confirmed:
            if measurement.value <= self.exit_threshold:
                self._confirmed = False
                self._consecutive_count = 0
                self._last_transition = "exit"
            return self._confirmed

        if measurement.value >= self.enter_threshold:
            self._consecutive_count += 1
            if self._consecutive_count >= self.confirmation_samples:
                self._confirmed = True
                self._last_transition = "enter"
        else:
            self._consecutive_count = 0
        return self._confirmed

    def reset(self) -> None:
        self._consecutive_count = 0
        self._confirmed = False
        self._last_transition = None


class AdaptiveHysteresisRadiationTrigger:
    """Raise the entry threshold once after each completed radiation episode.

    The planner hazard reference remains fixed.  Only the trigger threshold is
    adapted, starting at ``initial_entry_multiplier * base_hazard_reference``
    and increasing by that amount after each RADIATION-to-NORMAL exit, capped
    at the base reference.  The exit threshold is always derived from the
    current episode's entry threshold using the configured hysteresis margin.
    """

    def __init__(
        self,
        *,
        base_hazard_reference: float,
        initial_entry_multiplier: float,
        quantity: str,
        unit: str,
        hysteresis_margin: float = NOMINAL_HYSTERESIS_MARGIN,
        confirmation_samples: int = NOISE_FREE_CONFIRMATION_SAMPLES,
    ) -> None:
        if not math.isfinite(base_hazard_reference) or base_hazard_reference <= 0.0:
            raise ValueError("base_hazard_reference must be finite and positive.")
        if not math.isfinite(initial_entry_multiplier) or not (
            0.0 < initial_entry_multiplier <= 1.0
        ):
            raise ValueError(
                "initial_entry_multiplier must lie in the interval (0, 1]."
            )
        if not math.isfinite(hysteresis_margin) or not (
            0.0 < hysteresis_margin < 1.0
        ):
            raise ValueError("hysteresis_margin must lie strictly between 0 and 1.")
        if isinstance(confirmation_samples, bool) or not isinstance(
            confirmation_samples,
            int,
        ):
            raise TypeError("confirmation_samples must be an integer.")
        if confirmation_samples <= 0:
            raise ValueError("confirmation_samples must be positive.")
        if not quantity or not unit:
            raise ValueError("quantity and unit must be non-empty.")

        self.base_hazard_reference = float(base_hazard_reference)
        self.initial_entry_multiplier = float(initial_entry_multiplier)
        self.hysteresis_margin = float(hysteresis_margin)
        self.confirmation_samples = confirmation_samples
        self.quantity = quantity
        self.unit = unit
        self._initial_entry_threshold = (
            self.initial_entry_multiplier * self.base_hazard_reference
        )
        self._consecutive_count = 0
        self._confirmed = False
        self._last_transition: str | None = None
        self._completed_episode_count = 0
        self._threshold_increment_count = 0
        self._current_entry_threshold = self._initial_entry_threshold
        self._last_observation_episode_index = 1
        self._last_observation_entry_threshold = self._current_entry_threshold
        self._last_observation_exit_threshold = self.current_exit_threshold

    @property
    def consecutive_count(self) -> int:
        return self._consecutive_count

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    @property
    def last_transition(self) -> str | None:
        return self._last_transition

    @property
    def initial_entry_threshold(self) -> float:
        return self._initial_entry_threshold

    @property
    def current_entry_threshold(self) -> float:
        return self._current_entry_threshold

    @property
    def current_exit_threshold(self) -> float:
        return (1.0 - self.hysteresis_margin) * self._current_entry_threshold

    @property
    def enter_threshold(self) -> float:
        """Compatibility alias for the currently armed entry threshold."""
        return self.current_entry_threshold

    @property
    def exit_threshold(self) -> float:
        """Compatibility alias for the current episode's exit threshold."""
        return self.current_exit_threshold

    @property
    def current_episode_index(self) -> int:
        """Return the one-based index of the current or next episode."""
        return self._completed_episode_count + 1

    @property
    def completed_episode_count(self) -> int:
        return self._completed_episode_count

    @property
    def threshold_increment_count(self) -> int:
        """Return how often a completed exit numerically raised the threshold."""
        return self._threshold_increment_count

    @property
    def last_observation_episode_index(self) -> int:
        return self._last_observation_episode_index

    @property
    def last_observation_entry_threshold(self) -> float:
        return self._last_observation_entry_threshold

    @property
    def last_observation_exit_threshold(self) -> float:
        return self._last_observation_exit_threshold

    def observe(self, measurement: RadiationMeasurement) -> bool:
        """Consume one measurement and adapt only after a completed exit."""
        if not isinstance(measurement, RadiationMeasurement):
            raise TypeError("measurement must be a RadiationMeasurement.")
        if measurement.quantity != self.quantity or measurement.unit != self.unit:
            raise ValueError(
                "Measurement quantity/unit does not match the hazard contract."
            )

        episode_index = self.current_episode_index
        entry_threshold = self.current_entry_threshold
        exit_threshold = self.current_exit_threshold
        self._last_observation_episode_index = episode_index
        self._last_observation_entry_threshold = entry_threshold
        self._last_observation_exit_threshold = exit_threshold
        self._last_transition = None

        if self._confirmed:
            if measurement.value <= exit_threshold:
                self._confirmed = False
                self._consecutive_count = 0
                self._last_transition = "exit"
                self._completed_episode_count += 1
                next_threshold = min(
                    (self._completed_episode_count + 1)
                    * self._initial_entry_threshold,
                    self.base_hazard_reference,
                )
                if next_threshold > self._current_entry_threshold:
                    self._threshold_increment_count += 1
                self._current_entry_threshold = next_threshold
            return self._confirmed

        if measurement.value >= entry_threshold:
            self._consecutive_count += 1
            if self._consecutive_count >= self.confirmation_samples:
                self._confirmed = True
                self._last_transition = "enter"
        else:
            self._consecutive_count = 0
        return self._confirmed

    def reset_after_episode(self) -> None:
        """Re-arm transient state without losing mission-level adaptation."""
        self._consecutive_count = 0
        self._confirmed = False
        self._last_transition = None

    def reset(self) -> None:
        """Reset all adaptive state for a new complete mission."""
        self._consecutive_count = 0
        self._confirmed = False
        self._last_transition = None
        self._completed_episode_count = 0
        self._threshold_increment_count = 0
        self._current_entry_threshold = self._initial_entry_threshold
        self._last_observation_episode_index = 1
        self._last_observation_entry_threshold = self._current_entry_threshold
        self._last_observation_exit_threshold = self.current_exit_threshold


__all__ = [
    "AdaptiveHysteresisRadiationTrigger",
    "AboveBackgroundCriterion",
    "ConsecutiveRadiationTrigger",
    "HysteresisRadiationTrigger",
    "NOISE_FREE_CONFIRMATION_SAMPLES",
    "NOMINAL_HYSTERESIS_MARGIN",
    "RADIATION_CONFIRMATION_SAMPLES",
    "RadiationAnomalyCriterion",
]
