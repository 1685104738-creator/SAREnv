"""Data contracts shared by online radiation sensing and estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RadiationMeasurement:
    """One noise-free or noisy scalar observation at a simulated mission time."""

    x_m: float
    y_m: float
    altitude_m: float
    simulated_time_s: float
    value: float
    quantity: str
    unit: str

    def __post_init__(self) -> None:
        numeric = {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "altitude_m": self.altitude_m,
            "simulated_time_s": self.simulated_time_s,
            "value": self.value,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.altitude_m < 0.0:
            raise ValueError("altitude_m must be non-negative.")
        if self.simulated_time_s < 0.0:
            raise ValueError("simulated_time_s must be non-negative.")
        if self.value < 0.0:
            raise ValueError("Radiation measurement value must be non-negative.")
        if not self.quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not self.unit:
            raise ValueError("unit must be a non-empty string.")


@dataclass(frozen=True)
class RadiationEstimate:
    """A planner-safe scalar estimate; ``None`` explicitly means unknown."""

    x_m: float
    y_m: float
    value: float | None
    weight_sum: float
    quantity: str
    unit: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("Estimate coordinates must be finite.")
        if not math.isfinite(self.weight_sum) or self.weight_sum < 0.0:
            raise ValueError("weight_sum must be finite and non-negative.")
        if self.value is None:
            if self.weight_sum != 0.0:
                raise ValueError("An unknown estimate must have zero weight_sum.")
        elif not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError("Known estimate value must be finite and non-negative.")
        elif self.weight_sum <= 0.0:
            raise ValueError("A known estimate must have positive weight_sum.")
        if not self.quantity:
            raise ValueError("quantity must be a non-empty string.")
        if not self.unit:
            raise ValueError("unit must be a non-empty string.")

    @property
    def observed(self) -> bool:
        """Return whether this location has any estimator support."""
        return self.value is not None


__all__ = ["RadiationEstimate", "RadiationMeasurement"]
